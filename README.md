# MaskPLAN
**[MaskPLAN: Masked Generative Layout Planning from Partial Input](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MaskPLAN_Masked_Generative_Layout_Planning_from_Partial_Input_CVPR_2024_paper.html)**

*Hang Zhang, Anton Savov, Benjamin Dillenburger*

*Digital Building Technologies, ETH Zurich*

<img src='figures/CVPR_fig1_final.png' width=100%>

MaskPLAN allows users to influence layout generation with just the features they prioritize, using partial inputs in a Graph-structured Dynamic Masked Autoencoder (GDMAE), predicting layouts from incomplete design ideas.

[Paper Link](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MaskPLAN_Masked_Generative_Layout_Planning_from_Partial_Input_CVPR_2024_paper.html)

[Video Link](https://www.youtube.com/watch?v=HuzoJdMWnRM&t=5s)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/HuzoJdMWnRM&t=5s/hqdefault.jpg)](https://www.youtube.com/watch?v=HuzoJdMWnRM&t=5s)

## Installation
 
**1. install the requirements:**

Clone the repo and install conda env:
```
cd MaskPLAN
conda env create -f MaskPLAN.yaml
conda activate MaskPLAN
```

System requirements:
- a. GPU with at least 24GB Memory.
- b. System RAM at least 32GB.
- c. Hard-drive free storage at least 20GB.

**2. prepare the dataset:**

The original RPLAN dataset is from [RPLAN project page](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). Our data preprocess inherited the implementation from [RPLAN-Toolbox](https://github.com/zzilch/RPLAN-Toolbox) and [Graph2Plan-DataPreparation](https://github.com/HanHan55/Graph2plan/tree/master/DataPreparation). We've offered the pre-processed data in folder `Processed_data`. If you would like to walk through our data processing, follow the steps as follows:

- a. download the dataset from this [link](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform), put in [your_downloaded_path]. run through the *RPLAN_DataProcess.ipynb* in folder `Processed_data`, where the dataset path should follow [your_downloaded_path].

- b. pretraining the vqvae by run through the *VQVAE_Prior_Training_5564_B.ipynb* and *VQVAE_Prior_Training_5564_R.ipynb* in folder `VQVAE_Pretrain`. We've also offered our checkpoints, see folder `VQ_Pretrained`.

- c. run dynamic masking data processing in folder `Processed_data`. We offer the process of vector-only attributes representation (as ablation), and the img-vec hybrid attributes representation (as in our paper).

```
# data in vector-only format
python Processed_data/FP_DataProcess_vec.py

# data in vec-img hybrid format
python Processed_data/FP_DataProcess_cross.py
```

## training

We've offered our checkpoints in this [link](https://drive.google.com/drive/folders/1yvKe9l3l3zk7nM36LqgWmeeBIZTck0kp?usp=sharing) with four ablations (two model sizes with vector-only or vec-img hybrid data). If you would like to train the model, run the following code:

```
# training small mdoel in vector-only format
python Train/MaskPLAN_Train_Allvec.py

# training large mdoel in vector-only format
python Train/MaskPLAN_Train_AllvecDeep.py

# training small mdoel in vec-img hybrid format
python Train/MaskPLAN_Train_Allcross.py

# training large mdoel in vec-img hybrid format
python Train/MaskPLAN_Train_AllcrossDeep.py

```
## Inference

**1. fast test iteration:**

We offer faset test with random sampled iterations.

```
# Inference small mdoel in vector-only format
python Inference/MaskPLAN_Inference_iterate_vec_Single.py

# Inference large mdoel in vector-only format
python Inference/MaskPLAN_Inference_iterate_vec_Deep.py

# Inference small mdoel in vec-img hybrid format
python Inference/MaskPLAN_Inference_iterate_cross_Single.py

# Inference large mdoel in vec-img hybrid format
python Inference/MaskPLAN_Inference_iterate_cross_Deep.py

```

**2. UI interface:**

real-time UI interface.
- requirement: install Rhino 8.
- code: TODO

## Citation
```
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Hang and Savov, Anton and Dillenburger, Benjamin},
    title     = {MaskPLAN: Masked Generative Layout Planning from Partial Input},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8964-8973}
}
```
