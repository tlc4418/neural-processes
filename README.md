# Conditional Neural Processes: A Replication Study

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This codebase was created as part of a replication study concerning Conditional Neural Processes. It contains implementations of [Latent](https://arxiv.org/abs/1807.01622) and [Conditional Neural Processes](https://arxiv.org/abs/1807.01613), [Attentive Neural Processes](https://arxiv.org/abs/1901.05761), and [Convolutional Conditional Neural Processes](https://arxiv.org/abs/1910.13556). The work was conducted as part of the MLMI MPhil at the University of Cambridge.

We give a brief overview of the contents of the different repository folders:
- [data/](data/) contains classes to generate the 1D GP function data, 2D image data, and appropriate context generation needed to replicate experiments from the papers. 
- [models/](models/) contains implementations of the various models, including training and testing procedures. Some models even offer example scripts for training and visualizing the output. 
- [utils/](utils/) simply contains utility functions used throughout.