# MaskPLAN: Masked Generative Layout Planning from Partial Input

Hang Zhang, Anton Savov, Benjamin Dillenburger

Digital Building Technologies, ETH Zurich

## Accepted by CVPR 2024!

paper link: [https://github.com/HangZhangZ/MaskPLAN](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MaskPLAN_Masked_Generative_Layout_Planning_from_Partial_Input_CVPR_2024_paper.html)

## Abstract
 
Layout design in floorplan traditionally involves a labor-intensive iterative process for human designers. Recent advancements in generative modeling present a transformative potential to automate layout creation. However, prevalent models typically neglect crucial guidance from users, particularly their incomplete design ideas at the early stage. To address these limitations, we propose MaskPLAN, a novel user-guided generative design model formulated with Graph-structured Dynamic Masked AutoEncoders (GDMAE). Throughout its training phase, the layout attributes undergo stochastic masking to mimic partial input from users. During inference, layout designs are procedurally reconstructed via generative transformers. MaskPLAN incorporates the partial input as a global conditional prior, enabling users to turn incomplete design ideas into full layouts, which is a key part of real-world floorplan design. Notably, our proposed model offers an extensive range of adaptable user engagements, while also demonstrating superior performance to state-of-the-art methods in both quantitative and qualitative evaluations.

## Source Code and Data coming soon ...

### TODO

MaskPLAN Base Framework

MaskPLAN Ablations

Data Preprocessing Package

Preprocessed Dataset

Pre-trained Model

Inference GUI 
