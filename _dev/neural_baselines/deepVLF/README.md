# DeepVLF
We proposed deep variable-length feedback (DeepVLF) code, a novel DL-aided variable-length feedback coding scheme. By segmenting messages into multiple bit groups and employing a threshold-based decoding mechanism for independent decoding of each bit group across successive communication rounds, DeepVLF outperforms existing DL-based feedback codes and establishes a new benchmark in feedback channel coding.

## Requirements
A suitable conda environment named `DeepVLF` can be created and activated with:
```
conda env create -f environment.yaml
conda activate DeepVLF
```

## Train 
We implement a 2-stage strategy during the training process. Following command can be an example to train and evaluate a model for 1dB forward channel with noiseless feedback.
```python
python main.py --snr1 1 --snr2 100 --batchSize 8192 --totalbatch 10  --train 1 --core 1 --truncated 10 --restriction 'mid'
```

## Evaluate
Large batchsize could help improve the statistics accurancy in the evaluation stage. Following is an example for 1dB forward channel with noiseless feedback to evaluate our pretrained model's performance. Note that align `test_model` with `restriction`. 
```python
python main.py --snr1 1 --snr2 100 --batchSize 100000 --train 0 --truncated 10 --restriction 'mid' --test_model 'weights/weight_ff_1_fb_100_gamma_1-1e-5'
```
