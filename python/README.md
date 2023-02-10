# Python

This directory contains the python code used to perform estimation and simulation of the models detailed in the paper.

Please see notebooks `nb_Macro_data_(small)` and `nb_Macro_data_(medium)` for proper implementations of:

+ Data post-processing, splitting and normalization
+ Expanding and rolling window fit and forecasting
+ Forecast exporting for analyis in `R` (see `R` directory)
+ Plotting

## MFESN

Files `newToolbox_ESN.py` and `newToolbox_ESN_Multi.py` provide the classes and functions to work with ESN and MFESN models.

Suggested imports:

```python
from newToolbox_ESN import ESN, stateMatrixGenerator
from newToolbox_ESN_Multi import ESNMultiFrequency
```

### Model construction

Individual ESN models are constructed specifying parameters and random reservoir matrices, which we draw using the `stateMatrixGenerator()` function:

```python
esn_S_A = ESN(
    N=30,
    A=stateMatrixGenerator(
        (30, 30), 
        dist='sparse_normal', sparsity=10/30, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (30, K), 
        dist='sparse_uniform', sparsity=10/30, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=1,
    leak_rate=0.1,
    activation=np.tanh,
)
```

where `K` is the size of imputs.

MFESN models are composed of individual ESN model objects:

```python
esn_M_A = ESN(
    N=100,
    A=stateMatrixGenerator(
        (100, 100), 
        dist='sparse_normal', sparsity=10/100, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (100, K1), 
        dist='sparse_uniform', sparsity=10/100, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=1.5,
    leak_rate=0,
    activation=np.tanh,
)

esn_D_A = ESN(
    N=20,
    A=stateMatrixGenerator(
        (20, 20), 
        dist='sparse_normal', sparsity=10/20, normalize='eig',
        seed=20220623
    ),
    C=stateMatrixGenerator(
        (20, K2), 
        dist='sparse_uniform', sparsity=10/20, normalize='norm2',
        seed=20220623
    ),
    rho=0.5,
    gamma=0.5,
    leak_rate=0.1,
    activation=np.tanh,
)

esnMulti_A = ESNMultiFrequency((esn_M_A, esn_D_A), ar=False) 
```

### Fitting

Fitting of (MF)ESNs is implemented with ridge regression.

The ridge penalty is tunable using time-series-adapted cross-validation:

```python
cv_A = esnMulti_A.ridge_lambda_cv(
    Y=data_target, 
    z=(data_input_1, data_input_2),
    method="ridge-isotropic",
    cv_options="-cv:10-test_size:5",
    steps=1,
)
```

Fitting is immediate:

```python
esnMulti_A_fit = esnMulti_A.fit(
    Y=data_target, 
    z=(data_input_1, data_input_2), 
    method='ridge',
    Lambda=Lambda,
    full=False,
)
```

where `full=True` additionally implies the estimation of individual components coefficients (necessary for multi-step forecasting, c.f. paper).

### Forecasting

The main method for forecasting is given by `fixedParamsForecast()`:

```python
esnMulti_A_for = esnModel_A.fixedParamsForecast(
    Yf=data_target_test, 
    zf=(data_input_1_test, data_input_2_test), 
    fit=esnMulti_A_fit,
)
```

Expanding and rolling forecasting has been explicitly implemented in `nb_Macro_data_(small)` and `nb_Macro_data_(medium)` using only `fixedParamsForecast()`.

### Multi-step Forecasting

Multi-step (autonomous) MFESN forecasting can be done using `multistepForecast()` as below:

```python
esnModel.multistepForecast(
    Yf=data_target_test, 
    zf=(data_input_1_test, data_input_2_test),
    fit=esnMulti_fit,
    steps=4,
)
```

## MIDAS

File `newToolbox_MIDAS.py` provides the classes and functions to work MIDAS.

Suggested imports:

```python
from newToolbox_MIDAS import MIDAS
```

MIDAS model constructin requires to specify individual regresors lags and a `freqratio` tuple that also gives individual frequency ratios (of high-frequency observations to low-frequency obs.) for the model:

```python
Y_lags = 3
X_lags = tuple(9 for _ in range(K))

freqs = tuple(3 for _ in range(K))

MIDAS_model = MIDAS(
    freq=freqs, 
    ylags=Y_lags, 
    xlags=X_lags
)
```

Forecasting and fitting are implemented together in `MIDAS.fixedparamsForecast()`, so that an additional parameter `fit_length` is needed to slice the fitting sample from the testing sample:

```python
MIDAS_model.fixedparamsForecast(
    data=(data_target, data_regressors_as_list),
    fit_length=T,
    steps=1,
    grad=True,
    method='L-BFGS-B',
)
```

Note that `data_regressors_as_list` must be a list/tuple of *individual* regressors series e.g. one must slice a `pandas.DataFrame` into its individual columns.

Data preparation for MIDAS, as well as expanding and rolling forecasting implementations can be found in `nb_Macro_data_(small)` and `nb_Macro_data_(medium)`.

## DFM

The scripts `small_macro_data_dynamic_factor_model_macro.py` and `medium_dynamic_factor_model_macro.py` implement DFM model specifications and forecasting.

Pease note that these are very computationally intesive and should be run on a computing cluster. Notebooks `nb_Macro_data_(small)` and `nb_Macro_data_(medium)` load pre-computed forecasts as well to keep execution time feasible.
