# SAMCT
This repo is the official implementation for:\
[SAMCT: Segment Any CT Allowing Labor-Free Task-Indicator Prompts.](https://arxiv.org/pdf/2403.13258.pdf)\
(The details of our SAMCT can be found in the models directory in this repo or the paper.)

## Highlights
🏆 SAMCT supports two modes: interactive segmentation and automatic segmentation. (allowing both manual prompts and labor-free task-indicator prompts)\
🏆 CT5M is a large CT dataset. (about 1.1M images and 5M masks covering 118 categories)\
🏆 Excellent performance. (superior to both foundation models and task-specific models)

## Installation
Following [Segment Anything](https://github.com/facebookresearch/segment-anything) and [SAMUS](https://github.com/xianlin7/SAMUS), `python=3.8.16`, `pytorch=1.8.0`, and `torchvision=0.9.0` are used in SAMCT.

1. Clone the repository.
    ```
    git clone https://github.com/xianlin7/SAMCT.git
    cd SAMCT
    ```
2. Create a virtual environment for SAMCT and activate the environment.
    ```
    conda create -n SAMCT python=3.8
    conda activate SAMCT
    ```
3. Install Pytorch [`pytorch=1.8.0`] and TorchVision [`torchvision=0.9.0`].
   (you can follow the instructions [here](https://pytorch.org/get-started/locally/))
5. Install other dependencies.
  ```
    pip install -r requirements.txt
  ```
(* If you have already installed our SAMUS, you can skip steps 2-4 above, and activate the environment of SAMUS `conda activate SAMUS`)
## Checkpoints
- We use checkpoint of SAM in [`vit_b`](https://github.com/facebookresearch/segment-anything) version during training SAMCT.
- The checkpoint of SAMCT trained on CT5M can be downloaded [here](https://drive.google.com/file/d/13YlRjVlsWv4OrAdC349g5uC7vbW7ajXC/view?usp=sharing).

## Data
- CT5M consists of 30 publicly-available datasets:
    - [Hemorrhage-BrainCTImages](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images)
    - [INSTANCE](https://instance.grand-challenge.org)
    - [HaN-Seg](https://han-seg2023.grand-challenge.org/)
    - [Totalsegmentator](https://zenodo.org/record/6802614#.ZGcjznZBy3B)
    - [HaN-OAR](https://structseg2019.grand-challenge.org/Dataset/)
    - [PDDCA](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)
    - [Covid-19-20](https://covid-segmentation.grand-challenge.org/COVID-19-20/)
    - [MosMed](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19)
    - [MM-WHS](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/)
    - [DecathlonZ Task06](http://medicaldecathlon.com/)
    - [DecathlonZ Task07](http://medicaldecathlon.com/)
    - [DecathlonZ Task09](http://medicaldecathlon.com/)
    - [DecathlonZ Task10](http://medicaldecathlon.com/)
    - [COVID-19 CT Scan](https://www.kaggle.com/datasets/andrewmvd/covid19-ct-scans)
    - [FUMPE](https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images)
    - [LCTSC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539#24284539e71bdc34fa4541fd9095ec534b094ed0)
    - [COVID-19 CT Image Seg](https://www.kaggle.com/competitions/covid-segmentation/data)
    - [LNDB](https://zenodo.org/record/7153205#.ZGYlGHZBy3C)
    - [ATM](https://atm22.grand-challenge.org/)
    - [Pediatric-CT-SEG](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096588#89096588dca4648279ca4dfc82b7c467d207a010)
    - [DecathlonZ Task08](http://medicaldecathlon.com/)
    - [AMOS22](https://amos22.grand-challenge.org/)
    - [WORD](https://github.com/hilab-git/word)
    - [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
    - [LiTS](https://competitions.codalab.org/competitions/17094)
    - [CHAOS](https://chaos.grand-challenge.org/)
    - [KiTS23](https://kits-challenge.org/kits23/)
    - [NIH-Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
    - [Adrenal-ACC-Ki67](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93257945)
    - [Prostate-Anatomical-Edge-Cases](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=145753229#1457532299ba0f6d7166845d88ea4d96f32f0c097)
- All images were saved in PNG format. No special pre-processed methods are used in data preparation.
- We have provided some examples to help you organize your data. Please refer to the file fold [example_of_required_dataset_format](https://github.com/xianlin7/SAMCT/tree/main/example_of_required_dataset_format).\
  Specifically, each line in all_train/all_val.txt should be formatted as follows:
  ```
    <class ID>/<dataset file folder name>/<image file name>
  ```
  (Here, "class ID" represents the label value of each category in the indicated dataset. For example, the "class ID" of the spleen, right kidney, left kidney, gallbladder, and esophagus on the BTCV dataset should be 1, 2, 3, 4, and 5, respectively.)
- The relevant information of your data should be configured in [./utils/config.py](https://github.com/xianlin7/SAMCT/blob/main/utils/config.py).
## Training
Once you have the data ready, you can start training the model.
```
cd "/home/...  .../SAMCT/"
python train.py --modelname SAMCT --task <your dataset config name>
python train_auto_prompt.py --modelname AutoSAMCT --task <your dataset config name>
```
## Testing
Do not forget to set the load_path in [./utils/config.py](https://github.com/xianlin7/SAMCT/blob/main/utils/config.py) before testing.
```
python testSAMCT.py --modelname SAMCT --task <your dataset config name>
python test.py --modelname AutoSAMCT --task <your dataset config name>
```

## Citation
If our SAMCT is helpful to you, please consider citing:
```
@misc{lin2024samct,
      title={SAMCT: Segment Any CT Allowing Labor-Free Task-Indicator Prompts}, 
      author={Xian Lin and Yangyang Xiang and Zhehao Wang and Kwang-Ting Cheng and Zengqiang Yan and Li Yu},
      year={2024},
      eprint={2403.13258},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
