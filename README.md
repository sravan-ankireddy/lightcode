# LightCode
Official Implementation of "[LightCode: Light Analytical and Neural Codes for Channels with Feedback](https://ieeexplore.ieee.org/abstract/document/10845797?casa_token=lbFfZEL3Q5EAAAAA:6Q1AVkK7kHGIxTzca6BtUxo8fMzjo6Gm1L-Zn7stNagEm76G3Z5RedmaUd1bV3irjf37tZNbsA)", IEEE Journal on Selected Areas in Communications, 2025.

## Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/sravan-ankireddy/lightcode.git
cd lightcode
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


Our implementation is largely based on "[generalized block attention feedback (GBAF) ](https://github.com/emre1925/GBAF)." We thank the authors for open sourcing their code. 
