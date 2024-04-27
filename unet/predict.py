import sys
import os
from pathlib import Path
from utils.config import Config
import tempfile
import shutil


def main(input_dir:Path, output_dir:Path, model_checkpoint_dir:Path, ncpus=1):
    model_name = "model-name"
    root_dir = Path(tempfile.mkdtemp(prefix="unet_"))
    shutil.copytree(model_checkpoint_dir, root_dir / "models" / model_name / "model/latest")
    raw_dir = root_dir / "Data/exp/raw"
    raw_dir.mkdir(parents=True)
    # make symlinks for all files within input_dir into raw_dir
    [os.symlink(f, raw_dir / f.name) for f in input_dir.iterdir() if f.is_file()]
    notes_run = f'predicting'

    c = Config(
        root=root_dir,
        datasource="exp",
        experiment=f'inference',
        train=False,
        model=model_name,
        predict_on='test',
        predictions_output_dir=output_dir,
    )
    
    c.num_cpu = ncpus
    c.print_recap(notes=notes_run)
    c.preprocess(create_csv=True)
    c.start(notes_run=notes_run)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))

# python predict.py /input /output /model 8