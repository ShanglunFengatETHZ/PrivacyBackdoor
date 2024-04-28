#  Analysis of fine-tuned models
Analyze the fine-tuned pre-trained models for data stealing. Compare initial weights and fine-tuned weights for reconstructing information from the fine-tuning process. Develop rules for displaying results.


### Reconstruct from MLP or ViT
Show reconstructed images, possible ground truth images. **Note** that if there are more than one possible ground truth images for a backdoor unit, we use gray panels instead.

```bash
python analysis/reconstruct_images.py --path ./weights/GROUP/NAME.pth --plot_mode recovery --arch vit --hw 4 8  --inches 4.35 2.15 --scaling 0.229 0.224 0.225 --bias 0.485 0.456 0.406
python analysis/reconstruct_images.py --path ./weights/GROUP/NAME.pth --plot_mode raw --arch vit --hw 4 8  --inches 4.35 2.15
```

`--path`: where to load the fine-tuned ViT

`--save_path`: [option] where to save the drawn pictures

`--plot_mode`: `recovery | raw | single`; `recovery`: show the reconstructed images; `raw`: show the possible ground truth images; `single`: show all possible ground truth for a backdoor unit

`--arch`: `toy | vit`; `toy`: apply to toy MLP model; `vit`: apply to ViT

`--hw`: [option] how many pictures are in each row and column while displaying 

`--inches`: [option] the size of the overall image size

`--scaling`, `--bias`: [option] $\hat{x} = a x + b$, $a$: scaling, $b$: bias, $x$:raw images $\hat{x}$: images to be presented. If normalized, scaling=(0.229 0.224 0.225), bias=(0.485 0.456 0.406). If not normalized, these parameters are unnecessary.

`--chw`: [option] reshape the image, used for the *recovery* mode and *toy* architecture

`--ids`: [option] present the possible ground truth images of a specific backdoor unit, only support the *toy* architecture currently



### Reconstruct from BERT
Show important information during reconstruction and save the reconstructed sentences and possible ground truth sentences. **Note** that we show possible ground truth sentences for a backdoor unit when there are not many, otherwise we show *MESS*, *NONE* instead.
* the most likely position index, and its cosine similarity
* the most likely word, and its cosine similarity
* the second most likely position index, and its cosine similarity
* the second most likely word, and its cosine similarity
* [option] how many ground truth sentences have activated a group of backdoor units.

```bash
python analysis/analyze_reconstruct_sentences.py --path ./weights/GROUP/NAME_monitor.pth --save_path ./text --verbose True
```

`--path`: where to load the information of MONITOR

`--save_path`: [option] where to save the ground truth sentences and reconstructed sentences

`--verbose`: [option] if `True`, print how many reasonable ground truth sentences for each backdoor unit.


### Gradient distribution of Differential Privacy
Show the distribution of gradients of one or two backdoor parameters during fine-tuning. We will see the gradient of a parameter concentrates on $0$ and $aC$.
We also display the mean and variance to illustrate how concentrated the gradient values are. We only show that a malicious MLP can cause concentrated gradients here, as our assumptions.
We do **NOT** provide further codes for a complete analysis of differential privacy.

```bash
python analysis/analyze_diffprv.py --path ./weights/GROUP/NAME_rgs_ex0.pth  --save_path ./pic --biconcentration True
```

`--path`: where to load weights from

`--save_path`: [option] where to save drawn pictures

`--biconcentration`: [option] if `True`, the gradient concentrates on two weights(used for the random head setting) instead of one weight(used for the MLP setting)


### Quality Analysis
Use PSNRs and SSIMs to show the quality of reconstruction.

```bash
python analysis/quality.py --path ./weights/20230918_complete/mlp_craftedhead_cifar10.pth --hw 8 8 --arch toy


python analysis/quality.py --path ./weights/20231228_spliceimage/vit_stitch_gelu_caltech.pth --hw 4 8 --step 4 

python analysis/quality.py --path ./weights/20230918_complete/vit_gelu_craftedhead_pet.pth --hw 8 8
```
    