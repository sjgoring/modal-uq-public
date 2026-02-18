
# modal-uq

Modular, config-driven codebase for **mode-centric** uncertainty quantification experiments:
Selective Prediction, OOD, and Active Learning.

## Quick start

# Ensure dependencies

py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt


# Running an experiment

```bash
python -m modal_uq.cli --config configs/selective_faithful.json
```

See `configs/` for runnable examples. Components are pluggable via a simple registry.


## Data sources

# Old faithful geyser data
[Todo summary of what the data represents]
As presented in the R package MASS. Note multiple versions of this data set are available. This is not the original, as durations noted as "short", "medium" or "long" have been replaced with imputed values.

References:
Azzalini, A. and Bowman, A. W. (1990) A look at some data on the Old Faithful geyser. Applied Statistics 39, 357–365.
Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with S. Fourth edition. Springer.

# Forest fires data
[Todo summary of what the data represents]

References:
Cortez, P. & Morais, A. (2007). Forest Fires [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D88D.
Cortez, P., & Morais, A. (2007). A data mining approach to predict forest fires using meteorological data.