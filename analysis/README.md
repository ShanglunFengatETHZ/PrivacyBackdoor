#  Analysis 

## Reconstruct from ViT
```bash
python analysis/reconstruct_images.py --path ./weights/GROUP/vit_relu_randhead_caltech.pth --plot_mode recovery --arch vit --hw 4 8  --inches 4.35 2.15 --scaling 0.229 0.224 0.225 --bias 0.485 0.456 0.406
python analysis/reconstruct_images.py --path ./weights/GROUP/vit_relu_randhead_caltech.pth --plot_mode raw --arch vit --hw 4 8  --inches 4.35 2.15
```

`--path`: where to load the fine-tuned ViT

`--save_path`: where to save the drawn pictures

`--plot_mode`: `recovery | raw | single`; `recovery`: show the reconstructed images; `raw`: show the possible ground truth images; `single`: show all possible ground truth for a backdoor unit

`--arch`: `toy | vit`; `toy`: apply to toy MLP model; `vit`: apply to ViT

`--hw`: how many pictures are in each row and column while displaying 

`--inches`: the size of the overall image size

`--scaling`, `--bias`: $\hat{x} = a x + b$, $a$: scaling, $b$: bias, $x$:raw images $\hat{x}$: images to be presented. If normalized and recovery mode, scaling=(0.229 0.224 0.225), bias=(0.485 0.456 0.406). If not normalized, these parameters are ununcessary

`--chw`: reshape the image, used when recovery mode for MLP

`--ids`: present the possible ground truth images of a specific backdoor unit, only for toy currently


## Reconstruct from BERT
```bash
python analysis/analyze_reconstruct_sentences.py --path ./weights/GROUP/bert_gelu_randhead_trec50_monitor.pth --save_path ./text --verbose True
```

`--path`: where to load the information of MONITOR

`--save_path`: where to save the ground truth sentences and reconstructed sentences

`--verbose`: if `True`, print how many reasonable ground truth sentences for each backdoor unit.


## Gradient distribution of Differential Privacy
```bash
python analysis/analyze_diffprv.py --path ./weights/20231005_diffprv/onlyprobe_epsilon3_rgs_ex0.pth  --save_path ./pic --biconcentration True
```

`--path`: where to load weights from

`--save_path`: where to save drawn pictures

`--biconcentration`: if `True`, the gradient concentrates on two weights(used for random head setting) instead of one weight(use for MLP setting)

    