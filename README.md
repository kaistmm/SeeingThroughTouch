# Seeing Through Touch: <br> Tactile-Driven Visual Localization of Material Regions
The official Pytorch implementation for "Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions", CVPR 2026 
<br> [Seongyu Kim](https://sites.google.com/view/seongyukim/), [Seungwoo Lee](https://mm.kaist.ac.kr/members/), [Hyeonggon Ryu](https://sites.google.com/view/mmmi-hufs/members/pi?authuser=0), [Joon Son Chung](https://mm.kaist.ac.kr/joon/), [Arda Senocak](https://ardasnck.github.io/)

<p align="center">
  <a href="https://mm.kaist.ac.kr/projects/SeeingThroughTouch/"><img src="https://img.shields.io/badge/STT-Project_Page-blue" alt="Project Page"></a>&nbsp;
  <a href="https://arxiv.org/abs/2604.11579"><img src="https://img.shields.io/badge/arXiv-2604.11579-b31b1b.svg" alt="arXiv"></a>&nbsp;
  <a href="https://huggingface.co/seongyu/SeeingThroughTouch/tree/main"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"></a>
</p>
<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="800"/>
</p>

## 1. Environment

```bash
# Create and activate conda environment
conda create -n stt python=3.10 -y
conda activate stt

# Install PyTorch
# The command below uses CUDA 11.8. Adjust the version according to your CUDA driver.
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
# --upgrade-strategy only-if-needed prevents other packages from overriding the installed PyTorch version.
pip install -r requirements.txt --upgrade-strategy only-if-needed

# Install KNN_CUDA
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Data Preparation

For dataset preparation and directory structure, please refer to *datasets/README.md*.

## 3. Evaluation

#### Checkpoint
Download the model checkpoints from the links and place them in the *checkpoints/* directory.
<table border="1">
  <tr>
    <td><b>Model</b></td>
    <td><b>Checkpoint</b></td>
  </tr>
  <tr>
    <td><code>STT-Local</code></td>
    <td><a href="https://huggingface.co/seongyu/SeeingThroughTouch/resolve/main/ckpt/STT-Local.pth">STT-Local.pth</a></td>
  </tr>
  <tr>
    <td><code>STT-Indomain</code></td>
    <td><a href="https://huggingface.co/seongyu/SeeingThroughTouch/resolve/main/ckpt/STT-Indomain.pth">STT-Indomain.pth</a></td>
  </tr>
  <tr>
    <td><code>SeeingThroughTouch (STT)</code></td>
    <td><a href="https://huggingface.co/seongyu/SeeingThroughTouch/resolve/main/ckpt/STT.pth">STT.pth</a></td>
  </tr>
</table>

#### TG-test
To evaluate on Touch and Go dataset, run:
```bash
bash shell/test_TG.sh
```

#### WebMaterial-test
To evaluate on WebMaterial dataset, run:
```bash
bash shell/test_WebMaterial.sh
```

#### OpenSurfaces-test
To evaluate on OpenSurfaces dataset, run:
```bash
bash shell/test_OS.sh
```

#### Interactive Localization
To evaluate on Interative Localization test set of WebMaterial, run:
```bash
bash shell/test_IIoU.sh
```


## 4. Training
Download [dinov3_vits16](https://github.com/facebookresearch/dinov3?tab=readme-ov-file) pretrained model onto your working server. <br>
To train the *Seeing Through Touch* model, run:
```bash
bash shell/train_STT.sh
```

## 5. Citation

If you find this work useful, please cite it as:

```bibtex
@inproceedings{kim2026seeingthroughtouch,
  author    = {Seongyu Kim and Seungwoo Lee and Hyeonggon Ryu and Joon Son Chung and Arda Senocak},
  title     = {Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## 6. Acknowledgments

This codebase is built upon [TVL (ICML2024)](https://github.com/Max-Fu/tvl). We thank the authors for their excellent work.