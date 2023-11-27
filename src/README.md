#  Malicious initialization \& fine-tuning

Initialize the MLP, ViT, BERT maliciously for attackers. Fine-tune the modified pre-trained models on down-stream tasks for the attacked.

You can run all procedures using a single command.
```bash
python src/main.py --mode vibkd --config_name vit_gelu_randhead_caltech
```

`--mode`: `mlpvn|vibkd|txbkd|stdtr|dpbkd`; `mlpvn`: a preliminary demonstration of the backdoor attack on a toy MLP model; 
`vibkd`: the backdoor attack on ViT; `txbkd`:the backdoor attack on BERT; `dpbkd`:the backdoor attack for differential privacy; `stdtr`: train a MLP model.

`--config_name`: path to the configuration file: `./experiments/configs/{CONFIG_NAME}.yml`. This configuration files is about the details of the malicious initialization and the fine-tuning recipe.

For different modes, the configuration templates are very different. We provide available templates of malicious ViT and BERT. 


### Source file explanation
Unlisted files are not used in the final version. In a sense, they are also interesting expansions.

`main.py`: manage files and modes.

`data.py`: cope with datasets

`train.py`: training and evaluation functions for different modes. In order to ensure the simplicity of a single function, it creates overall redundancy.

`tools.py`: Functions used in different files to handle some simple problems. Note that there are many functions not used in the final version. They are used in the preliminary version, which will not be presented.

*****
This part is about how to initialize models for data stealing. We construct baits in this part.

`model_mlp.py`: about all MLPs in this work. How to initialize a malicious MLP for reconstruction. How to initialize a malicious MLP for concentrating gradient.

⭐️ `edit_vit.py`: a **KEY** file you may be interested in. A wrapper of the vision transformer about malicious initialization and collecting important information. This is based on the architecture from Torchvision.

⭐️ `edit_bert.py`: a **KEY** file you may be interested in. A `monitor` to collect important information during fine-tuning. A function to initialize BERT. This is based on the architecture from Hugging Face.

*****
Conduct the training process from the configuration files for the data-stealing attack.

`run_mlp.py`, `run_vit.py`, `run_text_classification.py`

****
`run_dpprv.py`: Conduct the training process from the configuration files for DP-SGD black box attacks.
