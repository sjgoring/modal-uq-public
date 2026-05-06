
# modal-uq

Modular, config-driven codebase for **mode-centric** uncertainty quantification experiments:
Selective Prediction, OOD, and Active Learning.

## Quick start

## Ensure dependencies
Note if using BNNs Python<3.8 is required
3.7 ensures Python in the venv is v3.7.9 (long story as to why v3.7.9 is v3.7 on my machine)

## SG reminder, venv lives in temp to avoid file length issues.
py -3.7 -m venv .venv
._venv\Scripts\activate
py -m pip install -r requirements.txt

## Setup

Firstly, create a venv either with Python's venv system or Conda. This should be Python 3.10 for compatibility with the dependencies used in this project.

Next, install the requirements:

```bash
pip install -r requirements.txt
```

Then, install the modal-uq library locally so that the local dataset paths can be resolved. This is important!

```bash
# Make sure your terminal's current working directory is inside the root of the repository first!
pip install -e .
```

Now you're good to go!

## Running an experiment

```bash
python -m modal_uq.cli --config configs/selective_faithful.json
```

See `configs/` for runnable examples. Components are pluggable via a simple registry.


## Data sources

### Old faithful geyser data
[Todo summary of what the data represents]
As presented in the R package MASS. Note multiple versions of this data set are available. This is not the original, as durations noted as "short", "medium" or "long" have been replaced with imputed values.

References:
Azzalini, A. and Bowman, A. W. (1990) A look at some data on the Old Faithful geyser. Applied Statistics 39, 357–365.
Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with S. Fourth edition. Springer.

### Forest fires data
[Todo summary of what the data represents]

References:
Cortez, P. & Morais, A. (2007). Forest Fires [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D88D.
Cortez, P., & Morais, A. (2007). A data mining approach to predict forest fires using meteorological data.