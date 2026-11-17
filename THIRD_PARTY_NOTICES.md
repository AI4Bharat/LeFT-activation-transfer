# Third-party notices

The code in this repository is MIT-licensed (see `LICENSE`). It additionally
depends upon, reimplements or redistributes the components listed below.

## Dependencies

| Component                                  | Licence      |
| ------------------------------------------ | ------------ |
| PyTorch, torchvision                       | BSD-3-Clause |
| NumPy, SciPy, PyYAML, pandas, PyArrow      | BSD-3-Clause |
| Pillow                                     | MIT-CMU      |
| Hugging Face `datasets`, `huggingface-hub` | Apache-2.0   |
| pytest                                     | MIT          |

## Methods reimplemented here

* **PAU (Padé Activation Units)**: `src/activations/pade.py` is an independent
  implementation of the activation described in Molina, Schramowski and
  Kersting, *Padé Activation Units: End-to-end Learning of Flexible Activation
  Functions in Deep Networks* (ICLR 2020). No code from the authors' release is
  included.
* **Swin Transformer**: `src/models/swin.py` implements shifted-window
  attention following Liu et al. (ICCV 2021). It was written from the
  publication and is not derived from the Microsoft release.
* **MLP-Mixer**: `src/models/mlp_mixer.py` follows Tolstikhin et al.
  (NeurIPS 2021).
* **Linear attention**: `src/models/linvit.py` uses the elu+1 feature map of
  Katharopoulos et al., *Transformers are RNNs* (ICML 2020).

## Datasets

No dataset is redistributed in this repository. Fashion-MNIST (MIT),
CIFAR-10 and CIFAR-100 (MIT-style terms, University of Toronto), Food-101
(research-use terms, ETH Zurich) and ImageNet-1K (ImageNet terms of access,
non-commercial research) are obtained from their official sources by
`scripts/prepare_data.sh`. **The MIT licence on this code confers no rights
over any dataset.** The terms of each dataset should be consulted before
use.
