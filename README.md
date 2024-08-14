# LightCode
Code for "[LightCode: Light Analytical and Neural Codes for Channels with Feedback](https://arxiv.org/pdf/2403.10751)"

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/sravan-ankireddy/lightcode.git
cd deeppolar
```

Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Simulation settings

The code rate and the channel conditions cane be configured in parameters.py. Set 'train = 0' to run in inference mode.

## Training and Inference

```bash
python -u main.py
```

- `K`: block length
- `m`: modulation
- `-ell`: number of symbols
- `snr1`: forward channel snr
- `snr2`: feedback channel snr
- `arch`: choice of feature extractor
- `features`: choice of inputs to feature extractor
- `seed`: random seed (for reproducibility)
- `device` : cpu / 'cuda:0'
