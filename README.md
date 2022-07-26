# torch-styleddpm-vc

Torch implementation of StyleDDPM for voice conversion

- StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion, Li et al., 2021. [[arXiv:2107.10394](https://arxiv.org/abs/2107.10394)]
- VDM: Variational diffusion models, Kingma et al., 2021. [[arXiv:2107.00630](https://arxiv.org/abs/2107.00630)]
- UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models, Sasaki et al., 2021. [[arXiv:2104.05358](https://arxiv.org/abs/2104.05358)]
- MAE: Masked Autoencoders Are Scalable Vision Learners, He et al., 2021. [[arXiv:2111.06377](https://arxiv.org/abs/2111.06377)]
- StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN, Karras et al., 2019. [[arXiv:1912.04958](https://arxiv.org/abs/1912.04958)]


## Requirements

Tested in python 3.9.12 conda environment.

## Usage

Initialize the submodule and patch.

```bash
git submodule init --update
cd hifigan; patch -p0 < ../hifigan-diff
```

Download LibriTTS dataset from [openslr](https://openslr.org/60/)

Dump the preprocessed LibriTTS dataset.

```bash
python -m utils.vcdataset \
    --data-dir /datasets/LibriTTS/train-clean-360 \
    --out-dir /datasets/LibriTTS/train-clean-360-dump \
    --num-proc 8 \
    --chunksize 16 \
    --device cuda
```

To train model, run [train.py](./train.py)

```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360-dump \
    --from-dump
```
To start to train from previous checkpoint, --load-epoch is available.

```bash
python train.py \
    --data-dir /datasets/LibriTTS/train-clean-360-dump \
    --from-dump \
    --load-epoch 20 \
    --config ./ckpt/t1.json
```

Checkpoint will be written on TrainConfig.ckpt, tensorboard summary on TrainConfig.log.

```bash
tensorboard --logdir ./log
```

[TODO] To inference model, run [inference.py](./inference.py)

[TODO] Pretrained checkpoints are relased on [releases](https://github.com/revsic/torch-styleddpm-vc/releases).

To use pretrained model, download files and unzip it. Followings are sample script.

```py
from config import Config
from styleddpm import StyleDDPMVC

with open('t1.json') as f:
    config = Config.load(json.load(f))

ckpt = torch.load('t1_200.ckpt', map_location='cpu')

vc = StyleDDPMVC(config.model)
vc.load(ckpt)
vc.eval()
```

## [TODO] Learning curve


## [TODO] Figures

## [TODO] Samples
