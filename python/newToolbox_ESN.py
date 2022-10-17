#
# newToolbox_ESN
#
#   Basic ESN functionality
#
# Current version:      January 2022
# ================================================================

from multiprocessing.sharedctypes import Value
import types
from math import floor, ceil, inf 
import pandas as pd
import numpy as np
#from numpy import random
import re
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
from scipy.special import kl_div
from sklearn.model_selection import TimeSeriesSplit
from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.factory import get_termination

# ----------------------------------------------------------------
# Preamble

def identity(A):
    return A

def radbas(A):
    return np.exp(-A**2)

def retanh(A):
    return np.tanh(np.maximum(A, 0))

def softplus(A):
    return np.log(1 + np.exp(A))

def esn_data_to_nparray(data):
    v = None
    if (type(data) is pd.DataFrame):
        v = data.to_numpy(copy=True)
    elif (type(data) is pd.Series):
        v = data.to_numpy(copy=True)[:,None]
    elif type(data) is np.ndarray:
        v = np.copy(data)
    if (not v is None) and (v.ndim == 1):
        # Mutate to column vector
        v = np.atleast_2d(v)

    #print(v.shape)

    return v

def vech(V):
    assert type(V) is np.ndarray
    N1, N2 = np.atleast_2d(V).shape
    assert N1 == N2
    v = np.zeros(N1*(N1+1)//2)
    j = 0
    for n in range(N1):
        v[j:(j+N1-n)] = V[n:,n]
        j += N1-n
    return v

def matrh(v, N):
    assert type(v) is np.ndarray
    M = len(v)
    assert N*(N+1)//2 == M
    V = np.zeros((N, N))
    j = 0
    for n in range(N):
        V[n:,n] = v[j:(j+N-n)]
        j += N-n
    return V

# ----------------------------------------------------------------
# State map ingredients

# ----------------------------------------------------------------
# Fitting Methods

def ls_ridge(Y, X, Lambda=0):
    Tx, Kx = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    #print(X.shape)
    #print(Y.shape)

    try:
        Lambda = np.squeeze(np.asarray(Lambda))
    except:
        raise ValueError("Lambda must be a scalar or a 2D numpy array")
        
    # Regression matrices
    if Lambda.shape == ():
        # Lambda is a scalar 
        if Lambda < 1e-12:
            V = np.hstack((np.ones((Tx, 1)), X))
            res = np.linalg.lstsq(V, Y, rcond=None)
            W = res[0]
        else:
            W = np.linalg.solve(((X.T @ X / Tx) + Lambda * np.eye(Kx)), (X.T @ Y / Ty))
            #W = np.linalg.solve(((X.T @ X) + Lambda * np.eye(Kx)), (X.T @ Y))
            a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
            W = np.vstack((a, W))
    #elif 
    elif Lambda.shape == (Kx,Kx):
        W = np.linalg.solve(((X.T @ X / Tx) + Lambda), (X.T @ Y / Ty))
        #W = np.linalg.solve(((X.T @ X) + Lambda), (X.T @ Y))
        a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
        W = np.vstack((a, W))
    else:
        raise ValueError(f"Lambda is not scalar or 2D matrix with Gram shape ({Kx},{Kx}), found shape {Lambda.shape}")
    
    return W

def jack_ridge(Y, X, Lambda):
    Tx, _ = X.shape
    Ty, _  = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    if np.isscalar(Lambda):
        Lambda = Lambda * np.eye(X.shape[1])
    Gamma0 = ((X.T @ X / Tx) + Lambda)
    W = np.linalg.solve((Gamma0 @ Gamma0), (X.T @ Y / Ty))
    a = np.mean(Y.T, axis=1) - W.T @ np.mean(X.T, axis=1).T
    W = np.vstack((a, W))
    return W

def r_ls(Y, X, W0=None, P0=None):
    Tx, Kx = X.shape
    Ty, Ky = Y.shape
    assert Tx == Ty, "Shapes of X and Y non compatible"

    V = np.hstack((np.ones((Tx, 1)), X))

    if not P0 is None:
        assert P0.shape == (Kx+1, Kx+1)
        P = np.copy(P0)
    else:
        P = np.linalg.inv(V[[0],] @ V[[0],].T + np.eye(Kx+1))
        #P = np.linalg.inv(X.T @ X)

    if not W0 is None:
        assert W0.shape == (Kx+1, Ky)
        W = [np.copy(W0),]
    else:
        W = [np.zeros((Kx+1, Ky)),]
    Wt = np.copy(W[0])

    Yx = np.zeros(Y.shape)
    # Recursive least-squares updates
    for t in range(1, Ty):
        H = V[[t-1],]
        # Update
        K = P @ H.T @ np.linalg.pinv(1 + H @ P @ H.T)
        P = (np.eye(1+Kx) - K @ H) @ P
        
        Wt += (P @ H.T) @ (Y[t-1,] - H @ Wt)
        W.append(Wt)

        Yx[t,] = V[[t],] @ Wt

    return W, Yx  

# ----------------------------------------------------------------
# ESN Class
#

class ESN:
    def __init__(self, N=None, A=None, C=None, rho=0, zeta=None, gamma=1, leak_rate=0,
                    params=None, activation=identity):
        self.params_ = params
        self.activation_ = activation

        # Pre-allocations
        self.X_ = None
        #self.W_ = None

        # Parameter checks and allocation
        # number of neurons
        self.N_ = N #A.shape[0] if (not type(N) is int) else N
        # reservoir (connectivity) matrix 
        self.A_ = A
        # input mask        
        self.C_ = C
        # input scaling
        self.gamma_ = gamma
        # reservoir (connectivity) matrix spectral radius
        self.rho_ = rho
        # leak rate
        self.leak_rate_ = leak_rate * np.ones((self.N_, 1)) if (not type(leak_rate) is np.ndarray) else leak_rate
        # input shift
        self.zeta_ = zeta if (not zeta is None) else np.zeros((self.N_, 1))
        
        # If 'params' is set, overwrite ESN parameters:
        # useful to programmatically generate ESNs from tuple of parameters
        if not self.params_ is None:
            #A, C, gamma, zeta, rho, leak_rate = (None, None, 
            #                                     1, None, None, 0)
            if len(self.params_) == 3:
                A, C, rho = self.params_
            elif len(self.params_) == 4:
                A, C, rho, zeta = self.params_
            elif len(self.params_) == 5:
                A, C, rho, zeta, gamma = self.params_
            elif len(self.params_) == 6:
                A, C, rho, zeta, gamma, leak_rate = self.params_
            else:
                raise ValueError("Parameter tuple 'self.params_' has unknown content")

            # Assign
            self.N_ = N
            self.A_ = np.copy(A)
            self.C_ = np.copy(C)
            self.zeta_ = np.copy(zeta)
            self.gamma_ = np.copy(gamma)
            self.rho_ = np.copy(rho)
            self.leak_rate_ = np.copy(leak_rate)

        # Parameter checks
        N0, N1 = self.A_.shape
        assert N0 == N1, "A is not square"
        assert N0 == self.N_, "A does not have the size N"
        assert self.C_.shape[0] ==  N0, "A and C are not compatible"  
        assert self.zeta_.shape[0] == N0, "zeta is not shape-compatible with input"
        assert self.zeta_.shape[1] == 1, "zeta is not a vector"
        assert self.leak_rate_.shape[0] == N0, "Leak rate in not shape-compatible with states"
            
        assert np.isscalar(gamma), "gamma is not a scalar"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN State Gathering Functions
    # assuming indexing from 0
    #  feed z_0, z_1, ..., z_{T}

    def generate_states(self, z, A, C, rho, zeta, gamma, leak_rate,
                            init=None, collect='forecast', washout_len=0):
        # Flatten data
        z = np.atleast_2d(np.squeeze(z))
        #if z is None:
        #    raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray.")

        # Check dimensions
        if C.shape[1] != z.shape[0]:
            z = z.T
        Kz, T = z.shape
        assert Kz == C.shape[1], "C is not shape-compatible with input"

        # Gather states
        N = A.shape[0] 
        
        X = np.zeros((N, T))
        if init is None:
            init = np.zeros((N, 1))
        else:
            init = np.reshape(init, (N, 1))
        
        X[:, [0]] = (init * leak_rate + (1 - leak_rate) * self.activation_((A * rho) @ init
                                                                            + gamma * C @ z[:, [0]]
                                                                            + zeta))             
    
        for t in range(1, T):
            X[:, [t]] = (X[:, [t-1]] * leak_rate
                         + (1 - leak_rate) * self.activation_((A * rho) @ X[:, [t-1]]
                                                                + gamma * C @ z[:, [t]]
                                                                + zeta))
            
        X = X[:,washout_len:].T
        
        if collect == 'listen':
            self.X_ = X
        
        return X

    def base_generate_states(self, z, init=None, collect='forecast', washout_len=0):
        return self.generate_states(
            z=z, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_, 
                    gamma=self.gamma_, leak_rate=self.leak_rate_, 
                    init=init, washout_len=washout_len, collect=collect
        )

    def generate_autostates(self, T, W, A, C, rho, zeta, gamma, leak_rate, init):
        # Check T
        assert (type(T) is int) and (T > 0), "Number of autonomous states to generate 'T' must be a positive integer."
        
        # Gather autonomous-run states
        N = A.shape[0] 
        M = W.shape[1]
        
        y = np.zeros((M, T))
        X = np.zeros((N, T))
        if init is None:
            init = np.zeros((N, 1))
        else:
            init = np.reshape(init, (N, 1))

        # Initial state 
        X[:, [0]] = init  
        y[:, [0]] = W.T @ np.vstack((1, init))
    
        for t in range(1, T):
            X[:, [t]] = (
                X[:, [t-1]] * leak_rate + (1 - leak_rate) * 
                    self.activation_((A * rho) @ X[:, [t-1]] + gamma * C @ y[:, [t-1]] + zeta)
            )
            y[:, [t]] = W.T @ np.vstack((1, X[:,[t]]))

        return X.T, y.T

    def base_generate_autostates(self, T, W, init):
        return self.generate_autostates(
            T=T, W=W, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_,
                gamma=self.gamma_, leak_rate=self.leak_rate_,
                init=init
        )

    def filter_states_EKF(self, y, z, W, A, C, rho, zeta, gamma, leak_rate, 
                            Sigma_eps, Sigma_eta, mu_eps=None,
                            linearization='canonical', init=None, washout_len=0):
        # Flatten data
        #y = np.atleast_2d(np.squeeze(y))
        #z = np.atleast_2d(np.squeeze(z))

        N = A.shape[0] 
        #M = W.shape[1]

        W = np.atleast_2d(W)
        
        #print(W.shape)
        #print(y.shape)
        #print(z.shape)

        # Check dimensions
        if C.shape[1] != z.shape[0]:
            z = z.T
            y = y.T
        Ky, Ty = y.shape
        Kz, Tz = z.shape

        assert Ky == W.shape[1], "W is not shape-compatible with target"
        assert Kz == C.shape[1], "C is not shape-compatible with input"
        assert Ty == Tz, f"Target y and input z have different sample lengths: {Ty}, {Tz}"
        T = Ty

        # Init
        m0 = zeta
        P0 = 1e-2 * np.eye(N)
        if not init is None:
            m0, P0 = init
            assert m0.shape[0] == C.shape[0], "Init m0 is not shape-compatible with state space"
            assert P0.shape == A.shape, "Init P0 is not shape-compatible with state space"

        # Extended Kalman Filter
        m_t = m0
        P_t = P0

        #mu_eps = np.atleast_2d(mu_eps)
        Sigma_eps = np.atleast_2d(Sigma_eps)
        Sigma_eta = np.atleast_2d(Sigma_eta)

        W_a = W[0,:]
        W_w = W[1:,:]

        X_prd = np.zeros((N, T))
        X_flt = np.zeros((N, T))
        L_l_t = np.zeros(T)
        LogLike = 0

        if linearization == 'canonical':
            for t in range(T):
                u0_t = (rho * A) @ m_t + (gamma * C) @ z[:, [t]] + zeta
                m0_t = leak_rate * m_t + (1 - leak_rate) * np.tanh(u0_t)
                D_u0_t = 1 - (np.tanh(u0_t) ** 2)
                F_x = leak_rate * np.eye(N) + (1 - leak_rate) * D_u0_t * rho * A
                F_q = (1 - leak_rate) * D_u0_t * gamma * C
                P0_t = F_x @ P_t @ F_x.T + F_q @ Sigma_eps @ F_q.T
                P0_t = (P0_t + P0_t.T) / 2

                v_t = y[:, [t]] - (W_w.T @ m0_t + W_a[:,None])
                S_t = W_w.T @ P0_t @ W_w + Sigma_eta
                K_t = P0_t @ W_w @ np.linalg.inv(S_t)
                m_t = m0_t + K_t @ v_t
                P_t = P0_t - K_t @ S_t @ K_t.T
                P_t = (P_t + P_t.T) / 2
                if np.linalg.norm(P_t, ord=np.inf) <= 1e-11:
                    P_t = np.zeros((N, N))
                
                X_prd[:,t] = np.squeeze(m0_t)
                X_flt[:,t] = np.squeeze(m_t)
                L_l_t[t] = multivariate_normal.pdf(
                    np.squeeze(y[:, [t]]), 
                    mean=np.squeeze(W_w.T @ m0_t + W_a[:,None]), cov=S_t
                )
            #
            LogLike = np.sum(np.log(L_l_t[washout_len:] + 1e-12))
            
        elif linearization == 'joint':
            if np.linalg.norm(leak_rate) > 1e-9:
                raise ValueError("Joint EKF formulation is invalid for non-zero leak rates")

            for t in range(T):
                m0_t = (rho * A) @ np.tanh(m_t) + (gamma * C) @ z[:, [t]] + zeta
                F_x = (1 - (np.tanh(m_t) ** 2)).T * rho * A
                P0_t = F_x @ P_t @ F_x.T + (gamma ** 2) * (C @ Sigma_eps @ C.T)
                P0_t = (P0_t + P0_t.T) / 2

                v_t = y[:, [t]] - (W_w.T @  m0_t + W_a[:,None])
                H_x = W_w.T * (1 - (np.tanh(m0_t) ** 2)).T
                S_t = H_x @ P0_t @ H_x.T + Sigma_eta
                K_t = P0_t @ W_w @ np.linalg.inv(S_t)
                m_t = m0_t + K_t @ v_t
                P_t = P0_t - K_t @ S_t @ K_t.T
                P_t = (P_t + P_t.T) / 2
                if np.linalg.norm(P_t, ord=np.inf) <= 1e-11:
                    P_t = np.zeros((N, N))
                
                X_prd[:,t] = np.squeeze(m0_t)
                X_flt[:,t] = np.squeeze(m_t)
                L_l_t[t] = multivariate_normal.pdf(y[:, [t]], mean=(W_w.T @ m0_t + W_a[:,None]), cov=S_t)
            #
            LogLike = np.sum(np.log(L_l_t[washout_len:] + 1e-12))

        else:
            raise ValueError("EKF linearization type not recognized")
        
        return LogLike, X_prd, X_flt

    def base_filter_states_EKF(self, y, z, W, Sigma_eps, Sigma_eta, mu_eps=0,
                            linearization='canonical', init=None, washout_len=0):
        return self.filter_states_EKF(
            y=y, z=z, W=W, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_, gamma=self.gamma_, leak_rate=self.leak_rate_, 
                mu_eps=mu_eps, Sigma_eps=Sigma_eps, Sigma_eta=Sigma_eta,
                linearization=linearization, init=init, washout_len=washout_len
        )
        
    def base_fit_params_EKF(self, y, z, W, 
                            linearization='canonical', init=None, washout_len=0):
        
        Ny = y.shape[1]
        Nz = z.shape[1]

        def EKF_logLike(parsEKF):
            j = 0
            p_L_Sigma_eps = matrh(parsEKF[j:(j + Nz*(Nz+1)//2)], Nz)
            p_Sigma_eps = p_L_Sigma_eps @ p_L_Sigma_eps.T + 1e-6 * np.eye(Nz)
            j += Nz*(Nz+1)//2
            p_L_Sigma_eta = matrh(parsEKF[j:(j + Ny*(Ny+1)//2)], Ny)
            p_Sigma_eta = p_L_Sigma_eta @ p_L_Sigma_eta.T + 1e-6 * np.eye(Ny)
            # EKF filtering
            lL, X_prd, X_flt = self.base_filter_states_EKF(
                y=y, z=z, W=W, 
                mu_eps=None, Sigma_eps=p_Sigma_eps, Sigma_eta=p_Sigma_eta,
                linearization=linearization, init=init, washout_len=washout_len
            )
            return -lL

        # Optimize for Sigma_eps, Sigma_eta (given W)
        res = minimize(
            FunctionalProblem(
                (Nz*(Nz+1)//2 + Ny*(Ny+1)//2),
                EKF_logLike,
                x0 = np.hstack((1e-2 * vech(np.eye(Nz)), 1e-2 * vech(np.eye(Ny)))),
                xl = np.hstack((-1e1 * np.ones(Nz*(Nz+1)//2), -1e1 * np.ones(Ny*(Ny+1)//2))),
                xu = np.hstack((1e1 * np.ones(Nz*(Nz+1)//2), 1e1 * np.ones(Ny*(Ny+1)//2))),
            ), 
            PatternSearch(), 
            get_termination("time", "00:03:00"),
            #get_termination("time", "00:00:05"),
            verbose=True, 
            seed=1203477
        )

        # Unpack
        logLike = res.F
        j = 0
        L_Sigma_eps_opt = matrh((res.X)[j:(j + Nz*(Nz+1)//2)], Nz)
        Sigma_eps_opt = L_Sigma_eps_opt @ L_Sigma_eps_opt.T
        j += Nz*(Nz+1)//2
        L_Sigma_eta_opt = matrh((res.X)[j:(j + Ny*(Ny+1)//2)], Ny)
        Sigma_eta_opt = L_Sigma_eta_opt @ L_Sigma_eta_opt.T

        return (Sigma_eps_opt, Sigma_eta_opt, logLike)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Fitting
    # assuming indexing from 0
    # for forecasting feed z_1, z_2, ..., z_T and states have to be adjusted
    def fit(self, Y, z, method='ridge', Lambda=0, init=None, washout_len=0):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        # States
        X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=self.rho_, zeta=self.zeta_, 
                                    gamma=self.gamma_, leak_rate=self.leak_rate_, 
                                    init=init, washout_len=washout_len, collect="listen")

        # Fit
        W = None
        if (method == 'none') or (method is None):
            W = np.ones((X_.shape[1], 1))
            
        elif method == 'least_squares':
            W = ls_ridge(Y=Y_, X=X_, Lambda=0)

        elif method == 'ridge':
            W = ls_ridge(Y=Y_, X=X_, Lambda=Lambda)

        else:
            raise ValueError("Fitting method not defined")
        
        V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
        Y_fit     = V_fit @ W
        Residuals = Y_ - Y_fit
        RSS       = np.sum(Residuals ** 2)

        fit_out = {
            'model':        'ESN',
            'W':            W,
            'Y_fit':        Y_fit,
            'Residuals':    Residuals,
            'RSS':          RSS,
            'Y':            Y_,
            'V':            V_fit,
            'X':            X_,
            'method':       method,
            'Lambda':       Lambda,
            'init':         init,
            'washout_len':  washout_len,
        }

        return fit_out

    # Convenience fitting function for univariate time series
    def fit_ar(self, Y, method='ridge', Lambda=0, init=None, washout_len=0):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")

        Y0 = Y[:-1,]
        Y1 = Y[1:,]

        #print(Y)
        #print(Y0)
        #print(Y1)

        return self.fit(Y=Y1, z=Y0, method=method, Lambda=Lambda, init=init, washout_len=washout_len)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Forecasting
    # important that zf starts with t out of which is made a forecasting step
    # the output is step shifted w.r.t. zf
    def forecast(self, zf, fit, steps=1, method=None, init='cont'):
        # Flatten data
        zf_ = esn_data_to_nparray(zf)
        if zf_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Gather forecasting states
        if init is None:
            print("[ Forecasting states initialization is 'None': fallback to zero init ]\n")
            init_ = None
        # states contain X_0,...,X_{T}
        elif init == 'cont':
            init_ = fit['X'][-1,:]

        else:
            init_ = np.atleast_1d(init)
        
        Xf_ = self.base_generate_states(z=zf_, init=init_, collect='forecast')
        #Txf_, Nx, = Xf_.shape
        
        #assert Nx == self.N_, "dimension of generated states is incorrect"
        
        # Forecast
        Vf_ = None
        if (method == 'none') or (method is None):
            if steps == 1:
                Vf_ = np.hstack((np.ones((Xf_.shape[0]-1, 1)), Xf_[:-1,]))
            
        else:
            raise ValueError("Forecasting method not defined")

        Forecast = Vf_ @ fit['W']
        Yf_      = zf[1:,]

        forecast_out = {
            'Forecast':     Forecast,
            'Yf':           Yf_,
            'Vf':           Vf_,
            'Xf':           Xf_,
            'steps':        steps,
            'method':       method,
            'init':         init,
        }

        return forecast_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ESN Optimization
    def optim(self, Y, z, mode='rho_lambda', method='ridge', loss='RSS',
                Lambda=0, init=None, washout_len=0, esn_output=False, debug=False):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        # Decompose optimization type string
        mode_pars = re.findall("^\w+(?=-)?", mode)
        mode_opts = re.findall("(?<=-)\w+(?=:)?", mode)
        mode_nums = re.findall("(?<=:)(\d+(?:\.\d+)?)", mode)

        print(mode_pars)
        print(mode_opts)
        print(mode_nums)

        assert len(mode_pars) > 0, "Optimization method must be specified"
        mode_pars = mode_pars[0]


        # Define a fit function to make objective compact
        Wfun = None
        if method == 'least_squares':
            Wfun = lambda X, Y, L : ls_ridge(Y=Y, X=X, Lambda=0)

        elif method == 'ridge':
            Wfun = lambda X, Y, L : ls_ridge(Y=Y, X=X, Lambda=L)

        else:
            raise ValueError("Fitting method not defined")

        # Optimize
        res = None
        if mode_pars == 'rho_lambda':
            # Fallback parameters
            cv_splits = 0
            test_size = None
            batch_num = 1
            batch_var = 1e-2
            if 'cv' in mode_opts:    
                cv_splits = int(mode_nums[mode_opts.index('cv')])
                assert cv_splits >= 0
            if 'test_size' in mode_opts:
                test_size = int(mode_nums[mode_opts.index('test_size')])
                assert test_size >= 0
            if 'batch' in mode_opts:
                batch_num = int(mode_nums[mode_opts.index('batch')])
            if 'batch_var' in mode_opts:
                batch_var = float(mode_nums[mode_opts.index('batch_var')])

            # Generate splits
            if cv_splits > 2:
                tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)
                test_size = tscv.test_size
            else:
                tscv = None

            def HYPER_obj(pars2):
                p_rho, p_lambda = pars2
                p_lambda = 10**p_lambda

                # Batch
                A_list = [self.A_] #+ [self.A_ + np.random.standard_normal(size=self.A_.shape) * np.sqrt(batch_var) for _ in range(batch_num-1)]
                C_list = [self.C_] #+ [self.C_ + np.random.standard_normal(size=self.C_.shape) * np.sqrt(batch_var) for _ in range(batch_num-1)]

                # RSS
                RSS = 0
                for b in range(batch_num):
                    # States
                    X_ = self.generate_states(
                        z=z_, A=A_list[b], C=C_list[b], rho=p_rho, zeta=self.zeta_, 
                        gamma=self.gamma_, leak_rate=self.leak_rate_, 
                        init=init, washout_len=washout_len
                    )
                    if not tscv is None:
                        for train_index, test_index in tscv.split(X_):
                            W_ = Wfun(X_[train_index], Y_[train_index,], p_lambda)
                            Res = (Y_[test_index,] - np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_)
                            RSS += np.sum(Res ** 2)
                    else:
                        #W_ = Wfun(X_, Y_, p_lambda)
                        #Res = (Y_ - np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W_)
                        #RSS += np.sum(Res ** 2)
                        nMin = self.N_
                        for h in range(nMin, X_.shape[0]):
                            W_h = Wfun(X_[0:h,], Y_[0:h,], p_lambda)
                            Res_h = (Y_[h,] - np.hstack((1, X_[h,])) @ W_h)
                            RSS += np.sum(Res_h ** 2)
                #RSS /= batch_num * cv_splits * test_size

                return RSS

            #print(HYPER_obj([0.95, -4]))

            problem = FunctionalProblem(2,
                HYPER_obj,
                x0=[0.95, -4],
                xl=[0.01, -5],
                xu=[0.99, +5]
            )

            res_h = minimize(
                problem, 
                PatternSearch(n_sample_points=50), 
                #get_termination("n_eval", 500),
                get_termination("time", "00:01:00"),
                verbose=True, 
                seed=1203477
            )

            #print(res_h.X)

            # Evaluate optimization fit
            rho_opt = (res_h.X)[0]
            lambda_opt = 10**(res_h.X)[1]
            length_train = []
            W_ = []
            Y_fit = []
            Residuals = []

            if debug: 
                print("Best solution found:")
                print(f"rho        = {rho_opt}")
                print(f"lambda     = {lambda_opt}")

            X_ = self.generate_states(
                z=z_, A=self.A_, C=self.C_, rho=rho_opt, zeta=self.zeta_, 
                gamma=self.gamma_, leak_rate=self.leak_rate_,
                init=init, washout_len=washout_len
            )
            if not tscv is None:
                for train_index, test_index in tscv.split(X_):
                    W_split = Wfun(X_[train_index], Y_[train_index,], lambda_opt)
                    Y_fit_split = np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_split
                    Residuals_split = (Y_[test_index,] - Y_fit_split)
                    length_train.append(len(train_index))
                    W_.append(W_split)
                    Y_fit.append(Y_fit_split)
                    Residuals.append(Residuals_split)
            else:
                length_train = X_.shape[0]
                W_ = Wfun(X_, Y_, lambda_opt)
                Y_fit = np.hstack((np.ones((length_train, 1)), X_)) @ W_
                Residuals = (Y_ - Y_fit)

            # Output
            res = {
                'rho_opt':          rho_opt,
                'lambda_opt':       lambda_opt,
                'W':                W_,
                'Y_fit_opt':        Y_fit,
                'Residuals_opt':    Residuals,
                'Y':                Y_,
                'X':                X_,
                'length_train':     length_train,
                'x':                res_h.X,
                'fun':              res_h.F,
                'status':           "NA",
                'message':          "NA",
            }

            esn_opt = ESN(
                N=self.N_, A=self.A_, C=self.C_, activation=self.activation_,
                rho=rho_opt, gamma=self.gamma_, leak_rate=self.leak_rate_, 
            )

        elif mode_pars == 'RLGL':
            # Fallback parameters
            cv_splits = 0
            test_size = None
            if 'cv' in mode_opts:    
                cv_splits = int(mode_nums[mode_opts.index('cv')])
                assert cv_splits >= 0
            if 'test_size' in mode_opts:
                test_size = int(mode_nums[mode_opts.index('test_size')])
                assert test_size >= 0

            # Generate splits
            if mode_opts == 'cv':
                tscv = TimeSeriesSplit(n_splits=cv_splits)
            else:
                tscv = None

            def HYPER_obj(parsRLGL):
                p_rho, p_gamma, p_leak_rate, p_lambda = parsRLGL
                p_lambda = 10**(p_lambda)
                # States
                X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=p_rho, zeta=self.zeta_, 
                                            gamma=p_gamma, leak_rate=p_leak_rate, 
                                            init=init, washout_len=washout_len)
                Obj = 0
                if not tscv is None:
                    for train_index, test_index in tscv.split(X_):
                        W_ = Wfun(X_[train_index], Y_[train_index,], p_lambda)
                        if loss == 'RSS':
                            Res = (Y_[test_index,] - np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_)
                            Obj += np.sum(Res ** 2) 
                        elif loss == 'KL':
                            Obj += np.sum(kl_div(np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_ + 5, Y_[test_index,] + 5))
                else:
                    #W_ = Wfun(X_, Y_, p_lambda)
                    nMin = self.N_
                    for h in range(nMin, X_.shape[0]):
                        W_h = Wfun(X_[0:h,], Y_[0:h,], p_lambda)
                        #RSS += np.sum(Res_h ** 2)
                        if loss == 'RSS':
                            #Res = (Y_ - np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W_)
                            #Obj += np.sum(Res ** 2) 
                            Res_h = (Y_[h,] - np.hstack((1, X_[h,])) @ W_h)
                            Obj += np.sum(Res_h ** 2) 
                        elif loss == 'KL':
                            #Obj += np.sum(kl_div(np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W_ + 5, Y_ + 5))
                            Obj += 0

                return Obj

            problem = FunctionalProblem(
                4,
                HYPER_obj,
                x0=[0.95,  1,    0.1,   -4],
                xl=[0.01, 0.01,   0,    -5],
                xu=[2,     3,   0.99,   +5],
            )

            res_h = minimize(
                problem, 
                PatternSearch(), 
                #get_termination("n_eval", 50),
                get_termination("time", "00:03:00"),
                verbose=True, 
                seed=1203477
            )

            if debug:
                print("Best solution found: \nX = %s\nF = %s" % (res_h.X, res_h.F))

            # Evaluate optimization fit
            rho_opt = (res_h.X)[0]
            gamma_opt = (res_h.X)[1]
            leak_rate_opt = (res_h.X)[2]
            lambda_opt = 10**(res_h.X)[3]

            length_train = []
            W_ = []
            Y_fit = []
            Residuals = []

            X_ = self.generate_states(
                z=z_, A=self.A_, C=self.C_, rho=rho_opt, zeta=self.zeta_, 
                gamma=gamma_opt, leak_rate=leak_rate_opt, 
                init=init, washout_len=washout_len
            )
            if not tscv is None:
                for train_index, test_index in tscv.split(X_):
                    W_split = Wfun(X_[train_index], Y_[train_index,], lambda_opt)
                    Y_fit_split = np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_split
                    Residuals_split = (Y_[test_index,] - Y_fit_split)
                    length_train.append(len(train_index))
                    W_.append(W_split)
                    Y_fit.append(Y_fit_split)
                    Residuals.append(Residuals_split)
            else:
                length_train = X_.shape[0]
                W_ = Wfun(X_, Y_, lambda_opt)
                Y_fit = np.hstack((np.ones((length_train, 1)), X_)) @ W_
                Residuals = (Y_ - Y_fit)

            # Output
            res = {
                'rho_opt':          rho_opt,
                'gamma_opt':        gamma_opt,
                'leak_rate_opt':    leak_rate_opt,
                'lambda_opt':       lambda_opt,
                'W':                W_,
                'Y_fit_opt':        Y_fit,
                'Residuals_opt':    Residuals,
                'Y':                Y_,
                'X':                X_,
                'length_train':     length_train,
                'x':                res_h.X,
                'fun':              res_h.F,
                'status':           "NA",
                'message':          "NA",
            }

            esn_opt = ESN(
                N=self.N_, A=self.A_, C=self.C_, activation=self.activation_,
                rho=rho_opt, gamma=gamma_opt, leak_rate=leak_rate_opt, 
            )

        elif mode_pars == 'EKF':
            # Fallback parameters
            assert len(mode_opts) > 0, "EKF linearization option required, -canonical or -joint"
            linearization = mode_opts[0]

            Ny = Y_.shape[1]
            Nz = z_.shape[1]

            def HYPER_obj(parsEKF):
                p_lambda = parsEKF[0]
                p_rho = parsEKF[1]
                p_gamma = parsEKF[2]
                p_leak_rate = parsEKF[3]
                j = 3
                p_L_Sigma_eps = matrh(parsEKF[j:(j + Nz*(Nz+1)//2)], Nz)
                p_Sigma_eps = p_L_Sigma_eps @ p_L_Sigma_eps.T + 1e-6 * np.eye(Nz)
                j += Nz*(Nz+1)//2
                p_L_Sigma_eta = matrh(parsEKF[j:(j + Ny*(Ny+1)//2)], Ny)
                p_Sigma_eta = p_L_Sigma_eta @ p_L_Sigma_eta.T + 1e-6 * np.eye(Ny)
                # States
                X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=p_rho, zeta=self.zeta_, 
                                            gamma=p_gamma, leak_rate=p_leak_rate, 
                                            init=init, washout_len=washout_len)
                W_ = Wfun(X_, Y_, p_lambda)
                # EKF filtering
                LogLike, X_prd, X_flt = self.filter_states_EKF(
                    y=Y_, z=z_, W=W_, A=self.A_, C=self.C_, 
                        rho=p_rho, zeta=self.zeta_, gamma=p_gamma, leak_rate=p_leak_rate, 
                        mu_eps=np.zeros((self.N_, 1)), Sigma_eps=p_Sigma_eps, Sigma_eta=p_Sigma_eta,
                        linearization=linearization, init=init, washout_len=washout_len
                )

                return -LogLike

            problem = FunctionalProblem(
                (4 + Nz*(Nz+1)//2 + Ny*(Ny+1)//2),
                HYPER_obj,
                x0 = np.hstack((1e-5, 0.5, 1, 0, 1e-2 * vech(np.eye(Nz)), 1e-2 * vech(np.eye(Ny)))),
                xl = np.hstack((1e-7, 0.001, 0, 0, -1e1 * np.ones(Nz*(Nz+1)//2), -1e1 * np.ones(Ny*(Ny+1)//2))),
                xu = np.hstack((1e2, 1.01, 1, 1, 1e1 * np.ones(Nz*(Nz+1)//2), 1e1 * np.ones(Ny*(Ny+1)//2))),
            )

            res_h = minimize(
                problem, 
                PatternSearch(), 
                get_termination("time", "00:03:00"),
                #get_termination("time", "00:00:05"),
                verbose=True, 
                seed=1203477
            )

            if debug:
                print("Best solution found: \nX = %s\nF = %s" % (res_h.X, res_h.F))

            # Evaluate optimization fit
            lambda_opt = (res_h.X)[0]
            rho_opt = (res_h.X)[1]
            gamma_opt = (res_h.X)[2]
            leak_rate_opt = (res_h.X)[3]
            j = 3
            L_Sigma_eps_opt = matrh((res_h.X)[j:(j + Nz*(Nz+1)//2)], Nz)
            Sigma_eps_opt = L_Sigma_eps_opt @ L_Sigma_eps_opt.T
            j += Nz*(Nz+1)//2
            L_Sigma_eta_opt = matrh((res_h.X)[j:(j + Ny*(Ny+1)//2)], Ny)
            Sigma_eta_opt = L_Sigma_eta_opt @ L_Sigma_eta_opt.T
            
            length_train = []
            W_ = []
            Y_fit = []
            Residuals = []

            X_ = self.generate_states(
                z=z_, A=self.A_, C=self.C_, rho=rho_opt, zeta=self.zeta_, 
                gamma=gamma_opt, leak_rate=leak_rate_opt, 
                init=init, washout_len=washout_len
            )

            length_train = X_.shape[0]
            W_ = Wfun(X_, Y_, lambda_opt)
            Y_fit = np.hstack((np.ones((length_train, 1)), X_)) @ W_
            Residuals = (Y_ - Y_fit)
                
            # Output
            res = {
                'lambda_opt':       lambda_opt,
                'rho_opt':          rho_opt,
                'gamma_opt':        gamma_opt,
                'leak_rate_opt':    leak_rate_opt,
                'Sigma_eps_opt':    Sigma_eps_opt,
                'Sigma_eta_opt':    Sigma_eta_opt,
                'W_opt':            W_,
                'Y_fit_opt':        Y_fit,
                'Residuals_opt':    Residuals,
                'Y':                Y_,
                'X':                X_,
                'length_train':     length_train,
                'x':                res_h.X,
                'fun':              res_h.F,
                'status':           "NA",
                'message':          "NA",
            }

            esn_opt = ESN(
                N=self.N_, A=self.A_, C=self.C_, activation=self.activation_,
                rho=rho_opt, gamma=gamma_opt, leak_rate=leak_rate_opt, 
            )

        elif mode_pars == 'ACsparsity_rho_lambda':
            # Fallback parameters
            batches   = 1
            cv_splits = 0
            if 'batch' in mode_opts:
                batches = int(mode_nums[mode_opts.index('batch')])
                assert batches > 0
            if 'cv' in mode_opts:
                cv_splits = int(mode_nums[mode_opts.index('cv')])
                assert cv_splits >= 0

            # Generate splits
            if cv_splits > 1:
                tscv = TimeSeriesSplit(n_splits=cv_splits)
            else:
                tscv = None

            # Check that A and C are defined as functions
            assert isinstance(self.A_, types.FunctionType)
            assert isinstance(self.C_, types.FunctionType)

            # TODO

        elif mode_pars == 'E_psi':
            # Fallback parameters
            batches   = 1
            cv_splits = 0
            if 'batch' in mode_opts:
                batches = int(mode_nums[mode_opts.index('batch')])
                assert batches > 0
            if 'cv' in mode_opts:
                cv_splits = int(mode_nums[mode_opts.index('cv')])
                assert cv_splits >= 0

            # Generate splits
            if cv_splits > 1:
                tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=1)
            else:
                tscv = None

            def HYPER_obj(parsPsi):
                p_psi = parsPsi[0]
                p_leak_rate = parsPsi[1]
                # Effective form
                p_rho = self.rho_ / self.gamma_
                p_zeta = self.zeta_ / self.gamma_
                # States
                X_ = self.generate_states(
                        z=z_, A=self.A_, C=self.C_, rho=(p_psi*p_rho), zeta=(p_psi*p_zeta), 
                        gamma=p_psi, leak_rate=p_leak_rate, 
                        init=init, washout_len=washout_len)
                Obj = 0
                if not tscv is None:
                    for train_index, test_index in tscv.split(X_):
                        W_ = Wfun(X_[train_index], Y_[train_index,], Lambda)
                        if loss == 'RSS':
                            Res = (Y_[test_index,] - np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_)
                            Obj += np.sum(Res ** 2) 
                        elif loss == 'KL':
                            Obj += np.sum(kl_div(np.hstack((np.ones((len(test_index), 1)), X_[test_index,])) @ W_ + 5, Y_[test_index,] + 5))
                else:
                    #W_ = Wfun(X_, Y_, Lambda)
                    #Res = (Y_ - np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W_)
                    #Obj += np.sum(Res ** 2) 
                    nMin = self.N_
                    for h in range(nMin, X_.shape[0]):
                        W_h = Wfun(X_[0:h,], Y_[0:h,], Lambda)
                        Res_h = (Y_[h,] - np.hstack((1, X_[h,])) @ W_h)
                        Obj += np.sum(Res_h ** 2) 

                return Obj

            #tmp_psi_ls = np.linspace(0.01, 2, 100)
            #tmp_CV_obj = np.array([HYPER_obj((l, 0)) for l in tmp_psi_ls])
            #plt.plot(tmp_psi_ls, tmp_CV_obj)
            #plt.grid()
            #plt.yscale('log')
            #plt.show()

            problem = FunctionalProblem(
                2,
                HYPER_obj,
                x0=[1, 0],
                xl=[0.01, 0],
                xu=[5, 1],
            )

            res_h = minimize(
                problem, 
                PatternSearch(), 
                get_termination("n_eval", 200),
                #get_termination("time", "00:03:00"),
                verbose=True, 
                seed=1203477
            )

            if debug:
                print("Best solution found: \nX = %s\nF = %s" % (res_h.X, res_h.F))

            # Evaluate optimization fit
            lambda_opt = Lambda
            psi_opt = (res_h.X)[0]
            rho_opt = psi_opt * (self.rho_ / self.gamma_)
            gamma_opt = psi_opt
            leak_rate_opt = (res_h.X)[1]
            zeta_opt = psi_opt * (self.zeta_ / self.gamma_)
            
            length_train = []
            W_ = []
            Y_fit = []
            Residuals = []

            X_ = self.generate_states(
                z=z_, A=self.A_, C=self.C_, rho=rho_opt, zeta=zeta_opt, 
                gamma=gamma_opt, leak_rate=leak_rate_opt, 
                init=init, washout_len=washout_len
            )

            length_train = X_.shape[0]
            W_ = Wfun(X_, Y_, lambda_opt)
            Y_fit = np.hstack((np.ones((length_train, 1)), X_)) @ W_
            Residuals = (Y_ - Y_fit)
                
            # Output
            res = {
                'lambda_opt':       lambda_opt,
                'psi_opt':          psi_opt,
                'rho_opt':          rho_opt,
                'gamma_opt':        gamma_opt,
                'leak_rate_opt':    leak_rate_opt,
                'zeta_opt':         zeta_opt,
                'W':                W_,
                'Y_fit_opt':        Y_fit,
                'Residuals_opt':    Residuals,
                'Y':                Y_,
                'X':                X_,
                'length_train':     length_train,
                'x':                res_h.X,
                'fun':              res_h.F,
                'status':           "NA",
                'message':          "NA",
            }

            esn_opt = ESN(
                N=self.N_, A=self.A_, C=self.C_, zeta=zeta_opt, activation=self.activation_,
                rho=rho_opt, gamma=gamma_opt, leak_rate=leak_rate_opt, 
            )

        else:
            raise ValueError("Optimization method descriptor not defined")

        if esn_output:
            return res, esn_opt
        else:
            return res

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting

    def plot_rho_lambda_loss(self, Y, z, vrho, vlambda, train_test=False, init=None, washout_len=0, figsize=(7, 6), surf3d=False):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        Nr = len(vrho)
        Nl = len(vlambda)

        # Move to log10 scale
        Xr, Yl = np.meshgrid(vrho, np.power(10, vlambda))
        
        def RSS_loss_surf(v):
            # States
            X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=v[0], zeta=self.zeta_, 
                                        gamma=self.gamma_, leak_rate=self.leak_rate_, 
                                        init=init, washout_len=washout_len, collect="listen")

            if train_test:
                train_split = floor(X_.shape[0] * 0.8)
                # Fit
                W         = ls_ridge(Y=Y_[0:train_split,], X=X_[0:train_split,], Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,]))
                Y_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,])) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_[train_split:,] - Y_fit) ** 2) / (X_.shape[0] - train_split)
            else:
                # Fit
                W         = ls_ridge(Y=Y_, X=X_, Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
                Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_ - Y_fit) ** 2) / X_.shape[0]
            
            return RSS
        
        RSS_loss_surf = Parallel(n_jobs=-1, verbose=False, prefer="processes")(
            delayed(RSS_loss_surf)(v) for v in zip(np.matrix.flatten(Xr), np.matrix.flatten(Yl))
        )

        # Reshape
        RSS_loss_surf = np.reshape(RSS_loss_surf, (Nr, Nl), order='F').T

        #RSS_loss_surf = np.full((Nr, Nl), np.nan)
        #for i, rho_i in enumerate(tqdm(vrho)):
        #    for j, lambda_i in enumerate(vlambda):
        #        # States
        #        X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=rho_i, zeta=self.zeta_, 
        #                                    gamma=self.gamma_, leak_rate=self.leak_rate_, 
        #                                    init=init, washout_len=washout_len, collect="listen")
        #        # Fit
        #        W = ls_ridge(Y=Y_, X=X_, Lambda=lambda_i)
        #        #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
        #        Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
        #        #Residuals = Y_ - Y_fit
        #        RSS       = np.sum((Y_ - Y_fit) ** 2)
        #        #
        #        RSS_loss_surf[j, i] = RSS

        # Plot
        if surf3d:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            scm1 = ax.plot_surface(Xr, np.log10(Yl), RSS_loss_surf, cmap=plt.cm.nipy_spectral, linewidth=0, antialiased=False)
            plt.colorbar(scm1)
        else:
            plt.figure(figsize=figsize)
            pcm1 = plt.pcolormesh(Xr, np.log10(Yl), RSS_loss_surf, cmap=plt.cm.nipy_spectral, shading='auto')
            plt.colorbar(pcm1)
        plt.xlabel('rho')
        plt.ylabel('log10(lambda)')
        plt.show()

        return RSS_loss_surf

    def plot_rho_gamma_loss(self, Y, z, vrho, vgamma, Lambda=0, log_xy=None, train_test=False, init=None, washout_len=0, figsize=(7, 6), surf3d=False):
        # Flatten data
        Y_ = esn_data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray")
        z_ = esn_data_to_nparray(z)
        if z_ is None:
            raise TypeError("Type of z not recognized, need pandas.DataFrame or numpy.ndarray")

        # Check sample size
        assert Y_.shape[0] ==  z_.shape[0], "Data inputs Y and z have incompatile sample sizes"

        Nr = len(vrho)
        Ng = len(vgamma)

        # Add log10 scale (if needed)
        vrho = np.power(10, vrho) if log_xy in ('x', 'xy') else vrho
        vgamma = np.power(10, vgamma) if log_xy in ('y', 'xy') else vgamma

        # Compute loss 
        Xr, Yg = np.meshgrid(vrho, vgamma)
        
        def RSS_loss_surf(v):
            # States
            X_ = self.generate_states(z=z_, A=self.A_, C=self.C_, rho=v[0], zeta=self.zeta_, 
                                        gamma=v[1], leak_rate=self.leak_rate_, 
                                        init=init, washout_len=washout_len, collect="listen")

            if train_test:
                train_split = floor(X_.shape[0] * 0.8)
                # Fit
                W         = ls_ridge(Y=Y_[0:train_split,], X=X_[0:train_split,], Lambda=Lambda)
                #V_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,]))
                Y_fit     = np.hstack((np.ones((X_[train_split:,].shape[0], 1)), X_[train_split:,])) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_[train_split:,] - Y_fit) ** 2) / (X_.shape[0] - train_split)
            else:
                # Fit
                W         = ls_ridge(Y=Y_, X=X_, Lambda=v[1])
                #V_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_))
                Y_fit     = np.hstack((np.ones((X_.shape[0], 1)), X_)) @ W
                #Residuals = Y_ - Y_fit
                RSS       = np.sum((Y_ - Y_fit) ** 2) / X_.shape[0]
            
            return RSS
        
        RSS_loss_surf = Parallel(n_jobs=-1, verbose=False, prefer="processes")(
            delayed(RSS_loss_surf)(v) for v in zip(np.matrix.flatten(Xr), np.matrix.flatten(Yg))
        )

        # Reshape
        RSS_loss_surf = np.reshape(RSS_loss_surf, (Nr, Ng), order='F').T

        # Remove log10 scale (if needed)
        Xr = np.log10(Xr) if log_xy in ('x', 'xy') else Xr
        Yg = np.log10(Yg) if log_xy in ('y', 'xy') else Yg

        # Plot
        if surf3d:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
            scm1 = ax.plot_surface(Xr, Yg, RSS_loss_surf, cmap=plt.cm.nipy_spectral, linewidth=0, antialiased=False)
            plt.colorbar(scm1)
        else:
            plt.figure(figsize=figsize)
            pcm1 = plt.pcolormesh(Xr, Yg, RSS_loss_surf, cmap=plt.cm.nipy_spectral, shading='auto')
            plt.colorbar(pcm1)
        plt.xlabel('rho')
        plt.ylabel('gamma')
        plt.show()

        return RSS_loss_surf


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helpers

def plotFitted(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    t = type(fit_out['Y'])
    h = fit_out['Y'].shape[1]
    if h > 1:
        gs = fig.add_gridspec(h, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        if (t is pd.DataFrame) or (t is pd.Series):
            for i, c in enumerate(fit_out['Y'].columns):
                axs[i].plot(fit_out['Y'][c], c='k', alpha=0.3, label=str(c))
                axs[i].plot(fit_out['Y_fit'][c], c='C'+str(i), label="Fitted")
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        else: 
            for i in range(h):
                axs[i].plot(fit_out['Y'][:,i], c='k', alpha=0.3, label=str(i))
                axs[i].plot(fit_out['Y_fit'][:,i], c='C'+str(i), label="Fitted")
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        axs[0].set_title("ESN - Fitted Values")
    else:
        plt.plot(fit_out['Y'], label="Data", alpha=0.3)
        plt.plot(fit_out['Y_fit'], label="Fitted")
        plt.grid()
        ax.legend()
        ax.set_title("ESN - Fitted Values")
    #return fig

def plotResiduals(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(fit_out['Residuals'], label="Residuals")
    plt.grid()
    ax.legend()
    ax.set_title("ESN - Fit Residuals")
    #return fig

def plotStates(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    N = fit_out['X'].shape[1]
    plt.plot(fit_out['X'][:,:min(N,100)], color='k', alpha=0.15)
    plt.grid()
    ax.set_title("ESN - Collected States [max 100]")

def plotW(fit_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(fit_out['W'], c="k", marker=".")
    plt.grid()
    ax.set_title("ESN - Estimated Weigths")
    #return fig

def plotOptimFitted(optim_out, figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    t = type(optim_out['Y'])
    h = optim_out['Y'].shape[1]
    r = np.arange(-1, len(optim_out['Y'])-1)
    if h > 1:
        gs = fig.add_gridspec(h, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        if (t is pd.DataFrame) or (t is pd.Series):
            for i, c in enumerate(optim_out['Y'].columns):
                axs[i].plot(r, optim_out['Y'][c], c='k', alpha=0.3, label=str(c))
                for j in range(len(optim_out['Y_fit_opt'])):
                    l1 = optim_out['length_train'][j]
                    l2 = optim_out['Y_fit_opt'][j].shape[0]
                    if l2 > 1:
                        axs[i].plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j][c], c='C'+str(i))
                    else:
                        axs[i].scatter(l1+l2, optim_out['Y_fit_opt'][j][c], c='C'+str(i))
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        else: 
            for i in range(h):
                axs[i].plot(r, optim_out['Y'][:,i], c='k', alpha=0.3, label=str(i))
                for j in range(len(optim_out['Y_fit_opt'])):
                    l1 = optim_out['length_train'][j]
                    l2 = optim_out['Y_fit_opt'][j].shape[0]
                    if l2 > 1:
                        axs[i].plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j][:,i], c='C'+str(i))
                    else:
                        axs[i].scatter(l1+l2, optim_out['Y_fit_opt'][j][:,i], c='C'+str(i))
                axs[i].grid()
                axs[i].legend()
                axs[i].label_outer()
        axs[0].set_title("ESN - Fitted Values")
    else:
        plt.plot(r, optim_out['Y'], label="Data", alpha=0.3)
        for j in range(len(optim_out['Y_fit_opt'])):
            l1 = optim_out['length_train'][j]
            l2 = optim_out['Y_fit_opt'][j].shape[0]
            if l2 > 1:
                plt.plot(np.arange(l1, l1+l2), optim_out['Y_fit_opt'][j])
            else:
                plt.scatter(l1, optim_out['Y_fit_opt'][j], marker='.')
        plt.grid()
        ax.legend()
        ax.set_title("ESN - Optimization - Fitted Values")
    #return fig

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# State Matrix Generator
    
def stateMatrixGenerator(shape, dist='normal', sparsity=None, normalize=None, options=None, seed=None):
    # stateMatrixGenerator
    #   Function to generate state matrices (e.g. state matrix A, input mask C)
    #   according to commonly used entry-wise or matrix-wise distributions.
    #
    #       shape       tuple, dimensions of the matrix to generate
    #       dist        type of matrix to generate
    #       sparsity    degree of sparsity (~ proportion of 0 elements)
    #                   of the generated matrix. Ignored if 'type' does
    #                   not have a 'sparse_' prefix
    #       normalize   normalization to apply to the matrix:
    #                       'eig'       maximum absolute eigenvalue
    #                       'sv'        maximum singular value
    #                       'norm2'     spectral norm
    #                       'normS'     infinity (sup) norm         
    #  

    # Set seed
    if not seed is None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(12345)

    if (shape is None) or (not type(shape) is tuple):
        raise ValueError("No shape selected")
    if (len(shape) > 2):
        raise ValueError("Shape tuple is larger than 2d")
    if dist is None:
        dist = 'normal'
    if sparsity is None:
        if re.findall('^sparse', dist):
            raise ValueError('Sparse distributions require choosing sparsity degree')
        sparsity = 1.
    else:
        if (sparsity < 0) or (sparsity >= 1):
            raise ValueError("Chosen sparsity degree is not within [0,1)")
    
    # Generate
    H = np.empty(shape)
    if dist == 'normal':
        H = rng.standard_normal(size=shape)
    elif dist == 'uniform':
        H = rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'sparse_normal':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.standard_normal(size=shape)
    elif dist == 'sparse_uniform':
        B = rng.binomial(n=1, p=sparsity, size=shape)
        H = B * rng.uniform(low=-1, high=1, size=shape)
    elif dist == 'orthogonal':
        H = np.linalg.svd(rng.standard_normal(size=shape))[0]
        # Ignore normalization
        normalize = None
    elif dist  == 'takens':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.eye((N1-1))
            H = np.hstack((H, np.zeros((N1-1, 1))))
            H = np.vstack((np.zeros((1, N1)), H))
            H = np.kron(np.eye(M), H)
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to takens")
    elif dist  == 'takens_exp':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            for j in range(1,M + 1):
                H = np.eye((N1-1))
                np.fill_diagonal(H, np.exp(-np.random.uniform(low=1, high=M, size=(1,1))*np.array(range(1,N1))))
                H = np.hstack((H, np.zeros((N1-1, 1))))
                H = np.vstack((np.zeros((1, N1)), H))
                if j == 1:
                    H_res = H
                else:
                    H_res = block_diag(H_res, H)
            H = H_res
        elif N1 > N2:
            H = np.hstack((1, np.zeros(N1-1)))
            H = np.atleast_2d(H).T
            id_matrix = np.eye(M)
            np.fill_diagonal(id_matrix, np.random.uniform(low=0, high=1, size=(1,M)))
            H = np.kron(id_matrix, H)
        else:
            raise ValueError("Shape not comformable to takens")
    elif dist == 'takens_augment':
        N1, N2 = shape
        M = 1
        if not options is None:
            M = options.get("M")
        if N1 == N2:
            H = np.zeros((M*N1, M*N1))
            for m in range(M):
                def underdiag_f(j):
                    #return np.random.uniform(low=0, high=0.5, size=(j))
                    #return (M-j)/(M+1) * np.ones((j))
                    #return np.exp(-(N1-1-j)) * np.ones((j))
                    return np.exp(-np.arange(N1-1, N1-1-j, -1)**.5)
                    #return np.exp(-np.arange(j)/N1)
                # Progressive fill under-diagonals
                H_m = np.atleast_2d(underdiag_f(1))
                for j in range(1, N1):
                    Q_j = np.zeros((j, j))
                    Q_j[1:,:-1] = H_m
                    np.fill_diagonal(Q_j, underdiag_f(j))
                    H_m = Q_j
                # Largest under-diagonal is just 1s (normalization)
                # Q_j = np.zeros((N1-1, N1-1))
                # Q_j[1:,:-1] = H_m
                # np.fill_diagonal(Q_j, np.ones(N1-1))
                H_m = Q_j
                H_m = np.hstack((H_m, np.zeros((N1-1, 1))))
                H_m = np.vstack((np.zeros((1, N1)), H_m))
                H[m*N1:(m+1)*N1,m*N1:(m+1)*N1] = H_m
        else:
            raise ValueError("Shape not comformable to takens")
    else:
        raise ValueError("Unknown matrix distribution/type")

    # Normalize
    if not normalize is None:
        if normalize == 'eig':
            H /= np.max(np.abs(np.linalg.eigvals(H)))
        elif normalize == 'sv':
            H /= np.max(np.linalg.svd(H)[1])
        elif normalize == 'norm2':
            H /= np.linalg.norm(H, ord=2)
        elif normalize == 'normS':
            H /= np.linalg.norm(H, ord=inf)
        else:
            raise ValueError("Unknown normalization")

    return H