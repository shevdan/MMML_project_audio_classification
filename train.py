import os
import torch
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], './audioset_tagging_cnn/pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], './audioset_tagging_cnn/utils'))
from main import train

DATASET_DIR = "./dataset_10"
WORKSPACE = "./dataset_10_1"

def plot_loss(train, val, workspace):
    plt.figure()
    sns.lineplot(train, label='Train loss')
    sns.lineplot(val, label='Val loss')
    plt.xlabel('Epochs')
    plt.savefig(f'./{workspace}/loss.png')
    

class Params:

    def __init__(self):
        self.workspace = WORKSPACE
        self.data_type = 'full_train'
        self.sample_rate = 22050
        self.window_size = 1024
        self.hop_size = 320
        self.mel_bins = 64
        self.fmin = 50
        self.fmax = 14000
        self.model_type = 'Cnn14'
        self.loss_type = 'clip_bce'
        self.balanced = 'balanced'
        self.augmentation = 'mixup'
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.resume_iteration = 0
        self.early_stop = 800
        self.cuda = True
        self.filename = 'main'


def transform_data():
    
    os.system(f'python utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path={DATASET_DIR}"/metadata/train.csv" \
    --audios_dir={DATASET_DIR}"/audios/train" \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/train.h5"')

    os.system(f'python utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path={DATASET_DIR}"/metadata/eval.csv" \
    --audios_dir={DATASET_DIR}"/audios/eval" \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/eval.h5"')

    os.system(f'python utils/dataset.py pack_waveforms_to_hdf5 \
    --csv_path={DATASET_DIR}"/metadata/test.csv" \
    --audios_dir={DATASET_DIR}"/audios/test" \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/test.h5"')

    print('---------------------------111---------------------------------')

def create_indexes():
    
    os.system(f'python utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/train.h5" \
    --indexes_hdf5_path={WORKSPACE}"/hdf5s/indexes/train.h5"')

    os.system(f'python utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/eval.h5" \
    --indexes_hdf5_path={WORKSPACE}"/hdf5s/indexes/eval.h5"')

    os.system(f'python utils/create_indexes.py create_indexes \
    --waveforms_hdf5_path={WORKSPACE}"/hdf5s/waveforms/test.h5" \
    --indexes_hdf5_path={WORKSPACE}"/hdf5s/indexes/test.h5"')

    print('---------------------------222---------------------------------')

def train_model():
    train_loss, eval_loss = train(Params())

    img = plot_loss(train_loss, eval_loss, WORKSPACE)
    
    
    # os.system(f"python pytorch/main.py train \
    # --workspace={WORKSPACE} \
    # --data_type=full_train \
    # --window_size=1024 \
    # --hop_size=320 \
    # --mel_bins=64 \
    # --fmin=50 \
    # --fmax=14000 \
    # --model_type=Cnn14 \
    # --loss_type=clip_bce \
    # --balanced=balanced \
    # --augmentation=mixup \
    # --batch_size=32 \
    # --learning_rate=1e-3 \
    # --resume_iteration=0 \
    # --early_stop=1500 \
    # --cuda")


   



if __name__ == "__main__":
    transform_data()
    create_indexes()
    train_model()