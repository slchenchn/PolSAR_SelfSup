## pipeline
1. train self-supervised model
2. convert model parameters using `benchmarks/detection/convert-pretrain-to-detectron2.py`
3. train segmentation model using mmseg