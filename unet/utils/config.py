# This scripts houses some usual config and constants used in the network
import os
import subprocess
import shutil
import sys
import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count
from socket import gethostname
from datetime import datetime
from .tic_toc import tic, toc

class ModelParameters():
    """ Class defining the models parameters
    """
    def __init__(self,
                 learning_rate = 1e-4,
                 num_epochs = 30,
                 bin_threshold = 0.7,
                 training_loss='WCE',
                 aug = 3,
                 segsize = (96, 96, 96)):
        # Number image per tfrecord in train set
        self.nTrainPerTfrecord = 1
        # Multi resolution patch size and spacing setting
        self.segsize = segsize
        self.patchsize_multi_res = [(1, segsize)] # [(resolution, patchsize)]
        self.test_patch_spacing = tuple(x//2 for x in segsize)
        # Visualization params
        self.num_images_to_show = 4
        # Training patch params
        self.num_pos = 30
        self.num_neg = 30
        # Number of augmentations
        self.aug = aug
        # Learning parameters
        self.batch_size = 12
        self.shuffle_buffer = 100
        self.learning_rate = learning_rate
        # Training parameters 
        self.num_epochs = num_epochs
        # Binarization threshold
        self.bin_threshold = bin_threshold
        # Training loss
        self.training_loss = training_loss
        self.loss_options = {'weight_alpha': 0.9}
    
    def print_recap(self):
        return
    


class Config(object):
    """
    Parameters
    ----------
        root: string, required
            Path to the directory root (subdivided into Data/ and models/)
        datasource : string or Path, required
            if string: self.root / "Data" / datasource
            if Path: datasource
            Directory name for the datasource of data.
        experiment: string, required
            Name of experiment (if train, will be the main model name). Needs to match the csv split file.
        train: boolean
            True for train, False for predict.
        model: string
            If train = False, this is the model name used to predict the data.
        tmp_storage: string
            Location of tfrecords. *None=in main exp directory. 'TMP'=$TMPDIR.
        which_model: string
            Chooses which model to load for predict. *'latest' -> latest | 'root' if checkpoint is at the root | 'your_name' -> this name.
        nickname: string
            Name of particular model or prediction directory. If already exists, append a timestamp to it (unless read_existing=True).
        read_existing: boolean
            *False. If set to True and new files are created, they might overwrite existing files. Example use includes prediction has already been run, and you want to access these to run further binarization or statistics.
        GPU: int, str, or list of. Default None
            Chooses which GPU(s) will be used for this session. If None, all available GPUs will be used. GPU=0 | GPU=[0,1] | GPU='1' accepted.
        predict_on: string
            Which images to run the prediction on without modifying the split file. *'test', 'all', 'train', 'val'.
        * = default value

    Main functions
    --------------
        self.print_recap():                     print a summary of run parameters to screen
        self.preprocess(force_preprocess):      preprocess images (resample, tf records for train/val)
        self.start():                           start the run (train or test)
        self.show_imgseg():                     return the command to run itksnap with image, segmentation and prediction
        self.binarize_and_stats(bin_threshold): binarize already existing predictions and compute stats
        self.lesionwise(bin_threshold):         compute lesion by lesion statistics
    """
    def __init__(self, root, datasource, experiment, train,
                 model_params=ModelParameters(),
                 model=None,
                 tmp_storage=None,
                 which_model='latest',
                 nickname='',
                 GPU=None,
                 continued_training=False,
                 read_existing=False,
                 predictions_output_dir=None,
                 predict_on='test'):
        self.GPU = self.set_GPU(GPU)
        self.host = gethostname()
        self.list_folders = []
        self.interactive = bool(getattr(sys, 'ps1', sys.flags.interactive))
        self.user = os.environ.get('USER') if os.environ.get('USER') else 'default'
        self.logging = {}
        self.type = 'training' if train else 'testing' # more precise definition further down
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if train==False and model is None:
            sys.exit("In testing mode, please define the model variable.")
        predict = not train
        self.train = train
        self.read_existing = read_existing
        self.num_cpu = self.set_CPU()
        self.model_params = model_params
        self.model_params.batch_size = self.set_batch_size()
        
        self.root = Path(root)
        self.add_dir(self.root)
        # The name of the specific run
        self.experiment = experiment
        # CSV file that stores train/test split
        self.train_test_csv = self.root / "models" / (self.experiment + ".csv") 

        # Directory where config.txt is saved
        if continued_training:
            self.main_dir = self.root / "models" / model
            self.experiment = model
        # fine-tuning conditions
        elif train and model is not None and model!=experiment:
            self.experiment = model + "." + experiment  # e.g. 293P.34U
            self.main_dir = self.root / "models" / self.experiment
        # normal training conditions, or further training of the same model
        elif train:
            self.main_dir = self.root / "models" / self.experiment
        # testing conditions
        else:
            self.main_dir = self.root / "models" / model
                

        ### IMAGES ###
        # Root directory for the images
        if isinstance(datasource, Path):
            data_dir = datasource
        else:
            data_dir = self.root / "Data" / datasource
        # Directory that contains the raw images and the segmented masks (patientid.nii.gz & patientid_seg.nii.gz)
        self.data_raw = data_dir / "raw"
        self.add_dir(self.data_raw) 
        # Directory that contains the preprocessed images 
        self.data_nh = data_dir / "preprocessed/noheader/"
        self.add_dir(self.data_nh)
        # Directory that contains the preprocessed images used for model training
        self.data_nh_1mm = data_dir / "preprocessed/noheader_1mm/"
        self.add_dir(self.data_nh_1mm)


        ### MODEL ###
        # Model
        if train:
            model_dir = self.root / "models" / self.experiment
            # Directory that contains the model
            if nickname:
                if (self.read_existing or not (model_dir / "model" / (nickname)).exists() or continued_training):
                    self.modelDir = model_dir / "model" / nickname
                else:
                    self.modelDir = model_dir / "model" / (nickname + '_' + self.timestamp)
            else:
                self.modelDir = model_dir / "model" / self.timestamp
            self.add_dir(self.modelDir)
            if (model and model!=experiment):
                self.type = 'fine-tuning'
            elif model:
                self.type = 'continued_training'
            else:
                self.type = 'training'
        else:
            model_dir = self.root / "models" / model
            # Directory that contains the model to load: 
            # will be the main directory, not one with the timestamps, 
            # so copy in main directory the model to use for predictions
            self.modelDir = self.choose_model_dir(model_dir / "model", which_model)
            # self.add_dir(self.modelDir) #TODO this should exist!
        # Load model
        if model is not None:
            self.load_model, self.starting_epoch = self.load_saved_model(model, which_model)
        else:
            self.load_model = None
            self.starting_epoch = 0
        # tfrecords
        if train:
            if tmp_storage:
                if tmp_storage=='TMP' and os.getenv('TMPDIR'):
                    tmp_dir = Path(os.getenv('TMPDIR'))
                elif tmp_storage=='scratch' and os.getenv('USER'):
                    tmp_dir = Path('/scratch') / os.getenv('USER') / 'tfrecords'
                else:
                    print(f'tmp_storage value is unknown, reverting to None behavior.\nYou passed {tmp_storage}.\nPass either "TMP" or "scratch".\n')
                    tmp_storage = None
            if tmp_storage and tmp_dir is not None:
                self.tfrDir = tmp_dir / experiment / "tf_records"
            else:
                self.tfrDir = model_dir / "tf_records"
            self.add_dir(self.tfrDir)
        # Tensorboard
        if train:
            self.tfboardDir = model_dir / "tfboard"
            self.add_dir(self.tfboardDir) 

        ### PREDICTIONS ###
        if predict:
            self.predict_on = predict_on
            if predictions_output_dir is None:
                if nickname:
                    if self.read_existing or not (model_dir / "predictions_output" / nickname).exists():
                        predictions_output_dir = model_dir / "predictions_output" / nickname
                    else:
                        predictions_output_dir = model_dir / "predictions_output" / (nickname + '_' + self.timestamp)
                else:
                    predictions_output_dir = model_dir / "predictions_output" / self.timestamp
            else:
                predictions_output_dir = Path(predictions_output_dir)
            self.add_dir(predictions_output_dir)
            # Model output data in:
            self.valoutDir = predictions_output_dir / "predictions"
            self.add_dir(self.valoutDir)
            # Resampled outptut data in:
            self.resampledDir = predictions_output_dir / "resampled_to_originalspacing"
            self.add_dir(self.resampledDir)
            # Thresholded outptut data in:
            self.thresholdedDir = predictions_output_dir / "binarized_masks"
            self.add_dir(self.thresholdedDir)
            # Dice scores and other analytics in:
            self.diceDir = predictions_output_dir / "Analytics"
            self.add_dir(self.diceDir)

        # disease to code
        self.diseaseCode = {
                "multiple_sclerosis_active": 0,
                "ADEM": 1,
                "adrenoleukodystrophy":2,
                "BG_normal":3,
                "normal":4,
                "Normal": 4, 
                "CADASIL":56,
                "CNS_lymphoma":5,
                "High_Grade_Glioma":6,
                "HIV_Encephalopathy":7,
                "Low_Grade_Glioma":8,
                "metastatic_disease":9,
                "Metastases": 9, 
                "migraine":10,
                "multiple_sclerosis_inactive":11,
                "neuromyelitis_optica":12,
                "PML":13,
                "PRES":14,
                "Susac_syndrome":15,
                "SVID":16,
                "toxic_leukoencephalopathy":17,
                "multiple_sclerosis_tumefactive":18,
                "vascular":19,
                "Hypoxic_Ischemic_Encephalopathy_acute":20,
                "Hypoxic_Ischemic_Encephalopathy_chronic":22,
                "Carbon_Monoxide_acute":23,
                "hemorrhage_chronic":24,
                "Hemorrhage_chronic":24,
                "Lymphoma":25,
                "Hemorrhage_subacute":26,
                "Nonketotic_Hyperglycemia":27,
                "Seizures":28,
                "Toxoplasmosis":29,
                "Cryptococcus":30,
                "Wilsons_Disease":31,
                "Artery_of_Percheron_acute":32,
                "infarct_chronic":33,
                "Infarct_chronic":33,
                "Deep_Vein_Thrombosis_subacute":34,
                "Deep_Vein_Thrombosis_chronic":35,
                "infarct_acute":36,
                "Infarct_acute":36,
                "Metastases":37,
                "Hypoxic_Ischemic_Encephalopathy_subacute":38,
                "Encephalitis":39,
                "Hemorrhage_chronic":41,
                "Deep_Vein_Thrombosis_acute":42,
                "Creutzfeldt_Jakob":43,
                "Wernicke_Encephalopathy":44,
                "Manganese_Deposition":45,
                "Carbon_Monoxide_subacute":46,
                "Calcium_Deposition":47,
                "Bilateral_Thalamic_Glioma":48,
                "Hemorrhage_acute":49,
                "Neuro_Behcet_Disease":50,
                "Sarcoidosis":51,
                "Neurofibromatosis":52,
                "Abscess":53,
                "infarct_subacute":54,
                "Infarct_subacute":54,
                "Carbon_Monoxide_chronic":55
                }
        
        # Elastic transform parameter
        self.ep = {
            'rotation_x': 0.001,
            'rotation_y': 0.001,
            'rotation_z': 0.05,
            'trans_x': 0.01,
            'trans_y': 0.01,
            'trans_z': 0.01,
            'scale_x': 0.1,
            'scale_y': 0.1,
            'scale_z': 0.1,
            'df_x': 0.1,
            'df_y': 0.1,
            'df_z': 0.1
        }
    

    ### HELPER FUNCTIONS ###
    def preprocess(self, force_preprocess=False, CV=None, create_csv=False):
        """ Run preprocessing for the images (resampling, train to tfrecords)
        Parameters
        ----------
            force_preprocess: boolean
                whether to force the reprocessing in case it finds that it had been done before
            CV: tuple (num_folds: int, test_size: float)
                whether to run cross-validation. Parameters: number of folds, test size (as a fraction - if None, set automatically)
            create_csv: boolean
                whether to create split csv
        """
        from image.process_image import strip_header_dir, resample_to_1mm_dir
        if create_csv:
            image_list = [im.name.split('.nii')[0] for im in self.data_raw.iterdir()
                if im.name.endswith(('.nii.gz', '.nii')) 
                    and not im.name.endswith(('_seg.nii.gz', '_seg.nii'))]
            n = len(image_list)
            _ = pd.DataFrame(
                {
                    'subject': image_list,
                    'condition': ['NA']*n,
                    'train_test': ['train' if self.train else 'test']*n
                }
            ).to_csv(self.train_test_csv, header=False, index=False)
        else:
            image_list = pd.read_csv(self.train_test_csv, header=None)[0].to_numpy()
        self.data_nh.mkdir(parents=True, exist_ok=True)
        self.data_nh_1mm.mkdir(parents=True, exist_ok=True)
        strip_header_dir(data_raw_dir=self.data_raw, data_nh_dir=self.data_nh, image_list=image_list, num_processes=self.num_cpu, force_preprocess=force_preprocess)
        resample_to_1mm_dir(data_nh_dir=self.data_nh, data_nh_1mm=self.data_nh_1mm, image_list=image_list, num_processes=self.num_cpu, force_preprocess=force_preprocess)
        if self.train:
            if CV:
                from .datasplit import create_cv_splits
                create_cv_splits(num_folds=CV[0], images_info_csv=self.train_test_csv, test_size=CV[1])
                self.model_params.nTrainPerTfrecord = 1
            preprocess = not self.check_for_tfrecords()
            if force_preprocess or preprocess:
                from image.preprocess_images import img_to_tfrecord
                # Creation of tfrecords
                self.make_dirs([self.tfrDir])
                img_to_tfrecord(self)
            if CV: print(f"Next step is to rerun Config with experiment set to {self.experiment}_k where k is the fold to run.")
        print('Preprocessing completed.')

    def start(self, notes_run='', freeze=None):
        """ Run training or testing
        Parameters
        ----------
            notes_run: string
                Add a few notes used in both the config.txt file for records (under `models/my_train_set/config.txt` or `predict/TRmy_train_set_TTmy_test_set`) and in tfboard name for easy filtering of experiments in Tensorboard
            freeze: string
                freeze all layers but those containing `freeze` in their name
        """
        self.make_dirs(self.list_folders)
        self.print_recap(notes=notes_run, print_=False, save=True)

        from model.train_test  import dataset_from_tfr, create_model, train, predict, predict_batch
        from .datasplit import img_list_from_split
        if self.train:
            # Load datasets
            img_list = img_list_from_split(self.train_test_csv, 'train') if self.model_params.nTrainPerTfrecord == 1 else None
            train_dataset = dataset_from_tfr(self.tfrDir, self.model_params.shuffle_buffer, self.model_params.batch_size, "train", img_list=img_list)
            img_list = img_list_from_split(self.train_test_csv, 'val') if self.model_params.nTrainPerTfrecord == 1 else None
            val_dataset = dataset_from_tfr(self.tfrDir, self.model_params.shuffle_buffer, self.model_params.batch_size, "val", img_list=img_list)
            # Train model
            train(train_dataset, self, notes=notes_run, validation_data=val_dataset, freeze=freeze)
        else:
            from image.process_image import resample_from_1mm_dir, binarize_dir
            from utils.calculateDice_MICCAI import compute_stats_dir
            # Run predictions
            predict_batch(self)
            # Process predictions
            resample_from_1mm_dir(self, num_processes=self.num_cpu)
            binarize_dir(self.resampledDir, self.thresholdedDir, threshold=self.model_params.bin_threshold, num_processes=self.num_cpu)
            # Compute Dice score
            stats_file_name = f'{self.diceDir}/stats_{self.experiment}_th_{self.model_params.bin_threshold}.csv'
            compute_stats_dir(self.data_raw, self.thresholdedDir / str(self.model_params.bin_threshold), stats_file_name, num_processes=self.num_cpu)
            print(stats_file_name)
        print("Done!")            

    def update_for_CV(self, fold:int):
        """Updates `self.train_test_csv` to match cross-validation experiment

        Args:
            fold (int): fold ID

        Raises:
            FileNotFoundError: no CV folds csv available
            FileNotFoundError: fold ID doesn't exist
        """
        new_csv = self.train_test_csv.with_name(
            f'{self.train_test_csv.stem}_{fold}.csv')
        if not new_csv.exists():
            folds = [csv.stem.split('_')[-1] for csv in self.train_test_csv.parent.iterdir()
                if csv.suffix == '.csv' and csv.name.startswith(self.train_test_csv.stem + '_')]
            if folds == []:
                raise FileNotFoundError(f'No CV folds csv available. Create one by running '
                    f'`c.preprocess(CV=(n_folds,None))`')
            raise FileNotFoundError(f'No csv found for {fold=}. Available {folds=}.')
        self.train_test_csv = new_csv

    def binarize_and_stats(self, bin_threshold):
        from image.process_image import binarize_dir
        from utils.calculateDice_MICCAI import compute_stats_dir
        binarize_dir(self.resampledDir, self.thresholdedDir, threshold=bin_threshold, num_processes=self.num_cpu)
        # Compute Dice score
        stats_file_name = f'{self.diceDir}/stats_{self.experiment}_th_{bin_threshold}.csv'
        stats = compute_stats_dir(self.data_raw, self.thresholdedDir / str(bin_threshold), stats_file_name, num_processes=self.num_cpu)
        print(stats_file_name)
        return stats

    def check_for_tfrecords(self):
        """ Returns True if tfrDir is not empty, False otherwise
        """
        try:
            n = len(list(self.tfrDir.iterdir()))
        except:
            print(f"The tfrecords directory is empty. Starting to preprocess.")
            return False
        if n == 0:
            print(f"The tfrecords directory is empty. Starting to preprocess.")
            return False
        else:
            return True

    def force_create(self, folder):
        """Recreate directory, deleting previous one"""
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
    
    def add_dir(self, folder):
        # if not self.read_existing:
        self.list_folders.append(folder)
    
    def make_dirs(self, list_folders):
        for folder in list_folders:
            folder.mkdir(parents=True, exist_ok=True)
    
    def splitfile_exist(self):
        if not os.path.isfile(self.train_test_csv):
            sys.exit(f"The split file does not exist at this location:\n{self.train_test_csv}")

    def choose_model_dir(self, model_dir, which_dir):
        if which_dir == 'latest':
            list_tmp = [directory for directory in model_dir.iterdir() if not directory.is_file()]
            latest_dir = max(list_tmp, key=os.path.getctime)
            # removes empty dir, until the last one
            while len(list_tmp)>1 and not any(latest_dir.iterdir()):
                list_tmp.remove(latest_dir)
                latest_dir = max(list_tmp, key=os.path.getctime)
            return latest_dir
        elif which_dir == 'root':
            return model_dir
        else:
            return model_dir / which_dir

    def load_saved_model(self, model, which_model):
        """ Find model checkpoint and corresponding epoch
        Returns:
            (latest_model, starting_epoch)
            where: latest_model=path(str), starting_epoch=int
        """
        from tensorflow.train import latest_checkpoint
        model_dir = self.choose_model_dir(self.root / "models" / model / "model", which_model)
        latest_model = latest_checkpoint(model_dir)
        # if no model is found, set load_model to False
        if latest_model is None:
            err_message = (
                f"There is an issue loading the model specified:\n"
                f"model name specified: {model}\n"
                f"which model specified: {which_model}\n"
                f"model directory inferred: {model_dir}\n"
                f"Check if the above directory has a `checkpoint` file, a `.cpkt.data` and `.ckpt.index`."
            )
            raise FileNotFoundErr(err_message)
        else:
            epoch = int(Path(latest_model).stem[3:]) # assuming the model name follows: cp-123.cpkt
            return latest_model, epoch

    def set_CPU(self, CPU:int=6):
        """Sets the number of CPUs to be used"""
        CPU_env = os.environ.get('SLURM_CPUS_ON_NODE')
        if CPU_env is not None:
            CPU_env = int(CPU_env)
            if CPU is not None:
                max_CPU = max(CPU,CPU_env)
                print(f'CPU set in config ({CPU}) and in environment ({CPU_env}) differ. Defaulting to {max_CPU}.')
                return max_CPU
            return CPU_env
        return CPU


    def set_GPU(self, GPU=None):
        """ Returns a tuple (num_gpus, gpu_names)
        """
        if GPU:
            if type(GPU) is str:
                GPU = list(map(int, GPU.split(',')))
            elif type(GPU) is int:
                GPU = [GPU]
            GPU_str = ','.join(list(map(str, GPU)))
            os.environ['CUDA_VISIBLE_DEVICES'] = GPU_str
        else:
            GPU_str = None
        from tensorflow.config.experimental import list_physical_devices, set_visible_devices, list_logical_devices, set_memory_growth
        gpus = list_physical_devices('GPU')
        if not gpus and GPU is not None:
            sys.exit(f"GPU specified: {GPU}. This device doesn't have any gpus.\n")
        elif not gpus and GPU is None:
            print("CPU job\n")
            # only using CPUs here
            return (0, 'CPU job')
        elif gpus and GPU is not None:
            if len(GPU) > len(gpus):
                sys.exit(f"GPU specified: {GPU}. Available GPUs: {list(range(len(gpus)))}. Please choose another GPU.\n")
        for gpu in gpus:
            set_memory_growth(gpu, True)
        if GPU_str:
            prettyprint = f"CUDA ({GPU_str}) - TF ({','.join([gpu.name for gpu in gpus])})"
        else:
            prettyprint = ','.join([gpu.name for gpu in gpus])
        num_gpu = len(gpus)
        return (num_gpu, prettyprint)

    def print_recap(self, notes="NA", print_=True, save=False):
        line = "-"*12
        recap_dic = {
                "Datetime:":                self.timestamp,
                "Experiment:":              self.experiment,
                "Type:":                    self.type,
                "Split file:":              self.train_test_csv,
                "Model dir:":               self.modelDir,
                "Data dir:":                self.data_nh_1mm,
                f"GPU ({self.GPU[0]}):":    self.GPU[1],
                "CPU usage:":               f"{self.num_cpu}/{cpu_count()}",
                "Host:":                    self.host,
                "User:":                    self.user,
                "Notes:":                   notes,
        }
        if self.type in ['fine-tuning', 'continued_training', 'testing']:
            recap_dic["Last checkpoint:"] = self.load_model
        if self.type == 'testing':
            recap_dic["Prediction dir:"] = self.valoutDir.parent
        if self.train:
            recap_dic["TfRecords:"]     = self.tfrDir
            recap_dic["Epochs:"]        = self.model_params.num_epochs
            recap_dic["Batch size:"]    = self.model_params.batch_size
            recap_dic["Learning rate:"] = self.model_params.learning_rate
        
        recap = f"\n{line*2} Parameters {line*2}\n"
        for key, value in recap_dic.items():
            recap += f"{key:<15}{value}\n"
        recap += f"{line*5}\n"

        if print_:
            print(recap)
        if save:
            with open(self.main_dir / "config.txt", 'a') as config_file:
                config_file.write(recap)

    def plot_logging(self):
        """ Return figure and pandas dataframe used to make it
        """
        if self.logging == {}: return
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        sns.set_theme()
        sns.set_context("paper")
        sns.set_style('whitegrid')
        times = pd.DataFrame({key: pd.Series(value) for key, value in self.logging.items()})
        colors = sns.color_palette('Set3', len(times.columns))
        pp = sns.boxplot(data=times, palette=colors);
        pp.set_title('Processing times');
        pp.set_ylabel('Time per iteration (s)');
        plt.yscale('log');
        plt.grid(which='major', linestyle='-');
        plt.grid(which='minor', linestyle='--');
        plt.xticks(rotation=30, horizontalalignment='right');
        return pp, times

    def show_imgseg(self, accession=None, no_true_seg=False, bin_threshold=None):
        """Return the command to load the original image, segmentation, and the prediction in itksnap. If no accession is provided, will pick one at random within prediction directory.
        """
        if bin_threshold is None: bin_threshold = self.model_params.bin_threshold
        if accession is None:
            import random
            # get random accession number
            accessions = [img.stem[:-4] for img in self.valoutDir.iterdir() if img.name.endswith('.nii.gz')]
            accession = random.choice(accessions)
        if no_true_seg:
            COMMAND = f'itksnap -g {self.data_raw}/{accession}.nii.gz -s {self.thresholdedDir}/{bin_threshold}/{accession}_binary.nii.gz'
        else:
            COMMAND = f'itksnap -g {self.data_raw}/{accession}.nii.gz -s {self.data_raw}/{accession}_seg.nii.gz -o {self.thresholdedDir}/{bin_threshold}/{accession}_binary.nii.gz'
        return COMMAND

    def set_batch_size(self):
        """Return automatically computed batch size to maximize GPU(s) usage.
        """
        if self.GPU[0] == 0: return 3           # if CPU job, default to batch size of 3
        gpu_memory = min(self.get_gpu_memory())
        num_GPUs = self.GPU[0]
        voxels = self.model_params.patchsize_multi_res[0][1][0] * self.model_params.patchsize_multi_res[0][1][1] * self.model_params.patchsize_multi_res[0][1][2]
        # Following formula
            # 0.80: observed that peak memory was capped somewhere between 75% and 89% (75% worked, the 89% went OOM)
            # 700 MiBs: weight of model, gradients, etc. (experimental)
            # 1.299e-3: slope, experimental
            # x0.5 if not using mixed precision (GPU Compute Capability < 7)
        optimal_batch_size_per_gpu = 0.80 * (gpu_memory - 700) / (1.299e-3 * voxels) # * 1.7
        if self.host in ['titan.radiology.ucsf.edu', 'cronus.radiology.ucsf.edu', 'titan', 'cronus']: optimal_batch_size_per_gpu = optimal_batch_size_per_gpu / 2
        optimal_batch_size = int(optimal_batch_size_per_gpu * num_GPUs)
        return optimal_batch_size


    def get_gpu_memory(self):
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    