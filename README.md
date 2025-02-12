# MaskPLAN: Masked Generative Layout Planning from Partial Input

Hang Zhang, Anton Savov, Benjamin Dillenburger

Digital Building Technologies, ETH Zurich

## Accepted by CVPR 2024!

paper link: [https://github.com/HangZhangZ/MaskPLAN](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MaskPLAN_Masked_Generative_Layout_Planning_from_Partial_Input_CVPR_2024_paper.html)

## Abstract
 
Layout planning spanning from architecture to interior design is a slow iterative exploration of ill-defined problems adopting a "I'll know it when I see it" approach to potential solutions. Recent advances in generative models promise to automate layout generation yet often overlook the crucial role of user-guided iteration cannot generate full solutions from incomplete design ideas and do not learn the inter-dependency of layout attributes. To address these limitations we propose MaskPLAN a novel generative model based on Graph-structured Dynamic Masked Autoencoders (GDMAE) featuring five transformers generating a blend of graph-based and image-based layout attributes. MaskPLAN lets users generate and adjust layouts with partial attribute definitions create alternatives for preferences and practice new composition-driven or functionality-driven workflows. Through cross-attribute learning and the user input as a global conditional prior we ensure that design synthesis is calibrated at every intermediate stage maintaining its feasibility and practicality. Extensive evaluations show MaskPLAN's superior performance over existing methods across multiple metrics.

## Source Code and Data coming soon ... _in several weeks_

### TODO

MaskPLAN Base Framework

MaskPLAN Ablations

Data Preprocessing Package

Preprocessed Dataset

Pre-trained Model

Inference GUI 

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
