# README
These configuration files are helpful when visitors cannot access the provided resources.

Since the pre-trained weights are in resources, visitors should at first use

```
python src/main.py --mode vibkd --config_name imagenet_small_gelu
```
for a useful small benign model. Then visitors can use the small benign pre-trained weights and 

```
python src/main.py --mode vibkd --config_name vit_gelu_randhead_caltech_splice
```
to conduct data-stealing attacks.