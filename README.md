# Optimal Sets and Solution Paths of ReLU Networks 

Code to replicate experiments in the paper [Optimal Sets and Solution Paths of
ReLU Networks](https://proceedings.mlr.press/v202/mishkin23a.html) by Aaron 
Mishkin and Mert Pilanci.

### Requirements

Python 3.8 or newer.

### Setup

Clone the repository using

```
git clone https://github.com/pilancilab/relu_optimal_sets.git
```

We provide a script for easy setup on Unix systems. Run the `setup.sh` file with

```
./setup.sh
```

This will:

1. Create a virtual environment in `.venv` and install the project dependencies.
2. Install `solfns` in development mode. This library contains infrastructure
 for running our experiments.
3. Create the `data`, `figures`, `tables`, and `results`  directories.

After running `setup.sh`, you need to activate the virtualenv using

```
source .venv/bin/activate
```

### Replications

The experiments are run via a command-line interface.
All experiments and plots/tables can be replicated with a single command.
First, make sure that the virtual environment is active.
Running `which python` in bash will show you where the active Python binaries are; 
this will point to a file in `relu_optimal_sets/.venv/bin` if the virtual 
environment is active.
Then, execute one of the files in the `scripts/` directory. 
Each file is named according to the figure or table in the paper which it 
reproduces.
For example, you can run the experiments and re-generate Figure 2 using,
```
python scripts/make_figure_2.py
```
The data for the experiments on CIFAR-10 and MNIST will be downloaded 
automatically, while the UCI datasets must be manually retrieved from 
[here](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz).


### Citation

Please cite our paper if you make use of our code or figures from our paper. 

```
@inproceedings{mishkin2023optimal,
  author       = {Aaron Mishkin and
                  Mert Pilanci},
  editor       = {Andreas Krause and
                  Emma Brunskill and
                  Kyunghyun Cho and
                  Barbara Engelhardt and
                  Sivan Sabato and
                  Jonathan Scarlett},
  title        = {Optimal Sets and Solution Paths of ReLU Networks},
  booktitle    = {International Conference on Machine Learning, {ICML} 2023, 23-29 July
                  2023, Honolulu, Hawaii, {USA}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {202},
  pages        = {24888--24924},
  publisher    = {{PMLR}},
  year         = {2023},
}
```

Looking for the poster for this paper?
See [relu optimal sets poster](https://github.com/aaronpmishkin/relu_optimal_sets_poster).

### Bugs or Other Issues

Please open an issue if you experience any bugs or have trouble replicating the experiments.
