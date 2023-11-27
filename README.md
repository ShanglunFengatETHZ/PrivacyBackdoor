# Backdoor Privacy

Manipulate weights to implant backdoors into a pre-trained model for a data-stealing attack.


*Example*: Reconstructed images and ground truth images of the malicious ViT fine-tuned on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset. We have successfully taken advantage of the [pre-trained weights](https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_32.html#torchvision.models.ViT_B_32_Weights) of ViT.

<img src=./materials/pics/vit_gelu_craftedhead_pet_reconstruction.svg width=40% /> <img src=./materials/pics/vit_gelu_craftedhead_pet_groundtruth.svg width=40% />

<img src=./materials/pics/acc_pet.svg width=60% /> 


Here are some [resources](https://drive.google.com/drive/folders/1QAjlQqNFK2ZOqly_CglapgLSs-hn0NP5?usp=sharing) about:
* configuration examples: malicious initializations & fine-tuning recipes
* additional pre-trained weights for transformers using ReLU or smaller transformers 
* some examples of the fine-tuned weights

<font color="darkred">**Note**</font>: we provide pre-trained weights of ViT and BERT using random heads for downstream classification tasks. It is possible that the pre-trained models break down during fine-tuning. Typically, breakdowns do not occur multiple times in succession. If the breakdown occurs this time, try training again. 
