#  README
This is about the source

```bash
python src/main.py --mode vibkd --config_name vit_gelu_randhead_caltech
```

`--mode`: `mlpvn|vibkd|txbkd|stdtr|dpbkd`; `mlpvn`: preliminary demonstration of the attack on toy MLP model; 
`vibkd`: the backdoor attack on ViT; `txbkd`:the backdoor attack on BERT  `dpbkd`:backdoor attack for differential privacy

`--config_name`: path to configuration = `./experiments/configs/{CONFIG_NAME}.yml`


### Source file explanation
`main.py`:

`data.py`:

`train.py`:

`tools.py`:

*****
`model_mlp`: 

⭐️ `edit_vit`:

⭐️ `edit_bert`:

*****
`run_mlp`:

`run_vit`:

`run_text_classification`:

****

`run_dpprv`:
