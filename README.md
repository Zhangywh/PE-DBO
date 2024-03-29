# PE-DBO

The code for the paper "High-Dimensional Dueling Optimization with Preference Embedding" published in AAAI 2023. 
The main paper and the appendix are in the `main.pdf` and `appendix.pdf` respectively, which are contained in the folder named `paper`.

## Requirements

Create the running environment with conda `4.10.3` with Python `3.9.0`: 

```shell
conda create -n pedbo python==3.9
conda activate pedbo
```

Necessary Python Libraries and versions: 

```shell
pip install bayesian-optimization==1.2.0
pip install cmaes==0.8.2
pip install botorch==0.6.0
pip install numpy==1.21.2
pip install pandas==1.3.4
```

## The Testing Functions and Datasets

### Official Links

- [Testing Functions](http://www.sfu.ca/~ssurjano/optimization.html)
- [MSLR Dataset](https://www.microsoft.com/en-us/research/project/mslr/)
- [ChEMBL Dataset](https://www.ebi.ac.uk/chembl/)

### Download and Use

#### Testing Functions

The testing functions ([Ackley](http://www.sfu.ca/~ssurjano/ackley.html),
[Dixon-Price](http://www.sfu.ca/~ssurjano/dixonpr.html),
[Levy](http://www.sfu.ca/~ssurjano/levy.html), and
[Sphere](http://www.sfu.ca/~ssurjano/spheref.html)) are encapsulated 
in the file `core.OF_TFunctions.py`. 

#### MSLR-WEB10K Dataset

Download the original dataset from [MSLR-WEB10K](https://1drv.ms/u/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78), 
and unzip the `MSLR-WEB10K` file and copy the files(`test.txt`, 
`train.txt`, `valid.txt`) in `Flod1` to MSLR folder of the project. 

#### ChEMBL Dataset

Use the dataset by calling
[design-bench](https://github.com/brandontrabucco/design-bench) 
Python library.

## Run Experiments

### Testing Functions Experiments and Scalability Experiments

Running our algorithm `PE-DBO.py` and the comparing experimental algorithms
(`PBO.py`, `KSS.py`, `COMP-UCB.py`) in the main folder. 

The hyperparameter setting method is as follows: 

```python
seed_start = 0  # random seed start
dim_high = 200  # the dimension of dataset / testing function
dim_eff = 10    # the effective dimension
dim_low = 12    # the dimension of low-dimension subspace in PEDBO
init_points = 30    # the initial duels number of optimization
low_bounds = 1.0    # the bounds of low-dimension subspace in PEDBO
exp_times = 20      # the number of repeat the experiment times
iteration_times = 50  # the number of iteration times every experiment
function_name = "ackley"  # the testing function name (only affects storage)

# the information of objective function (datasets or testing function)
objective_function = {
    'form': OF_TFunctions.ackley_function,  # choose the testing function
    'dim': dim_high,
    'bounds': BO.bounds11(dim_high),
    'optimal': 0
}

# the information of optimization dataset (subspace information)
dataset_info = {
    'dim': dim_low,
    'bounds': torch.tensor([[-low_bounds, low_bounds]] * dim_low)
}
```

Under the default setting, the testing functions experiment results of the paper 
can be reproduced. 

### Datasets Experiments

#### Effective Dimension

Run the file `eff_dim.py` in the folder `MSLR` and folder `ChEMBL` to get 
the experiment results of the effective dimension evaluation 
of the two datasets.

#### Train MSLR Dataset NN

As stated in the paper, we test the performance of different optimizers 
by training a NN for the MSLR dataset as the objective function of the 
optimizer. 

In `MSLR` folder, first run `PreProcess.py` to preprocess the data, then 
run `Fit.py` to get the NN model `model.pt`, 
and encapsulate the model `model.pt` into a callable optimization API 
through file `OF_Datasets.py` in folder `core`.

Call the real datasets objective functions in the same way as the testing
functions, except in file `OF_Datasets.py` not `OF_TFunctions.py`. 
After setting according to the appendix, the results in the paper 
can be reproduced. 

### Fixed Budget Experiments

Run files (`PE-DBO.py`, `PBO.py`, `REBO-cost.py` and `BO-cost.py`)
in the main folder to get the fixed budget experimental results. 
And when the objective function is set to be the MSLR dataset, 
the experimental results in the paper can be reproduced.

## The File Tree

```shell
.
├── core
│   ├── AF.py
│   ├── BO.py
│   ├── OF_Datasets.py
│   ├── OF_TFunctions.py
│   └── RE.py
├── MSLR
│   ├── PreProcess.py
│   ├── eff_dim.py
│   ├── fit.py
│   └── model.pt
├── ChEMBL
│   └── eff_dim.py
├── paper
│   ├── main.pdf
│   └── appendix.pdf
├── PE-DBO.py
├── PBO.py
├── KSS.py
├── COMP-UCB.py
├── BO-cost.py
├── REBO-cost.py
└── README.md
└── README.txt
```

Notes for the file tree:
- The files in the main folder reproduce the algorithm in this article and 5 other
comparison algorithms.
- The files in the folder `core` are the core components of the algorithms, 
which will be called directly from the various algorithms in the main folder.
- The files in the folder `MSLR` and `ChEMBL` are used in 
experiments on real datasets

## Reference
Zhang, Y.; Qian, H.; Shu, X.; and Zhou, A. "High-Dimensional Dueling Optimization with Preference Embedding." In Proceedings of
the 37th AAAI Conference on Artificial Intelligence (AAAI), 2023.
### Bibtex
```shell
@inproceedings{zhang2023preferenceem,
 author = {Zhang, Yangwenhui and Qian, Hong and Shu, Xiang and Zhou, Aimin},
 booktitle = {Proceedings of the 37th AAAI Conference on Artificial Intelligence},
 title = {High-Dimensional Dueling Optimization with Preference Embedding},
 year = {2023}
}
```
