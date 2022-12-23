#%% Imports
from utils.config import Config, ModelParameters

# Add a few notes
#       used in both the config.txt file for records (under `models/my_train_set/config.txt` or `predict/TRmy_train_set_TTmy_test_set`)
#       and in tfboard name for easy filtering of experiments in Tensorboard
notes_run = f"my_notes"
model_params = ModelParameters(learning_rate=1e-4, num_epochs=10)

### Choose *one* of the following Config call
#%% Config for normal training
c = Config(root='path/to/root/directory',
           datasource='my_image_data',
           experiment='my_train_set',
           train=True,
           nickname='my_unique_name',
           model_params=model_params,
           GPU='5,7',
           )
# Check that everything is setup correctly
c.print_recap(notes=notes_run)
# Start!
c.preprocess(force_preprocess=False)
c.start(notes_run=notes_run)

#%% Config for cross-validation training
fold = 1    # choose fold number
notes_run = f"fetal-brain-segmentation_fold-{fold}"
c = Config(root='path/to/root/directory',
           datasource='my_image_data',
           experiment='my_train_set',
           train=True,
           nickname=f'CV-{fold}',
           model_params=model_params,
           GPU='5,7',
           )
# Check that everything is setup correctly
c.update_for_CV(fold)
c.print_recap(notes=notes_run)
# c.preprocess(CV=(6,None), create_csv=True) # first time only, to automatically create folds
c.preprocess()

#%% Config for continued training
c = Config(root='', datasource='my_image_data',
            experiment='my_train_set',
            train=True,
            model='my_train_set',
            which_model='latest',
            nickname='my_unique_name',
            model_params=model_params,
            GPU='5,7')
# Check that everything is setup correctly
c.print_recap(notes=notes_run)
# Start!
c.preprocess(force_preprocess=False)
c.start(notes_run=notes_run)

#%% Config for testing
c = Config(root='', datasource='my_image_data',
            experiment='my_test_set',
            train=False,
            model='my_train_set',
            which_model='latest',
            nickname='my_unique_name_for_these_predictions',
            model_params=model_params,
            read_existing=False,
            GPU='5,7')
# Check that everything is setup correctly
c.print_recap(notes=notes_run)
c.preprocess(force_preprocess=False)
c.start(notes_run=notes_run)

# Get Main component from masks
import subprocess
original_images_dir = c.thresholdedDir / str(c.model_params.bin_threshold)
main_component_images_dir = original_images_dir.with_name(f'{c.model_params.bin_threshold}_MC')
main_component_images_dir.mkdir(exist_ok=True)
sp = subprocess.run(
    ['image/get_main_component.sh', original_images_dir, main_component_images_dir],
    capture_output=True,
    text='utf-8')
print(sp.stdout)
# Compute stats on main component masks
from utils.calculateDice_MICCAI import compute_stats_dir
stats_file_name = f'{c.diceDir}/stats_{c.experiment}_th_{c.model_params.bin_threshold}_MC.csv'
stats_df = compute_stats_dir(c.data_raw,
                             main_component_images_dir,
                             stats_file_name,
                             num_processes=12,
                             inference_only=False)
