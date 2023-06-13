#
# newToolbox_MIDAS
#
#   MIDAS regression
#
# Current version:      June 2022
#================================================================

from math import floor, ceil
import pandas as pd
import numpy as np
import scipy.optimize as optim
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize

# ----------------------------------------------------------------
# Preamble

def check_lags(lags_in):
    if isinstance(lags_in, int):
        assert (lags_in >= 0), "Scalar lag parameter must be integer and non-negative"
        lags = np.array((lags_in,)).astype(int)
    elif isinstance(lags_in, list) or isinstance(lags_in, tuple):
        lags = np.full(len(lags_in), np.nan).astype(int)
        for j in range(len(lags_in)):
            assert isinstance(lags_in[j], int) and (lags_in[j] >= 0), "All lag parameters must be integer and non-negative"
            lags[j] = lags_in[j]
    elif isinstance(lags_in, np.ndarray):
        lags = lags_in.astype(int)
        assert np.all(lags >= 0), "All lag parameters must be integer and non-negative"
    return lags

def check_freq(freq_in):
    if isinstance(freq_in, int):
        assert (freq_in > 0), "Scalar frequency must be integer and non-negative"
        freq = np.array((freq_in,)).astype(int)
    elif isinstance(freq_in, list) or isinstance(freq_in, tuple):
        freq = np.full(len(freq_in), np.nan).astype(int)
        for j in range(len(freq_in)):
            assert isinstance(freq_in[j], int) and (freq_in[j] >= 0), "All frequencies must be integer and non-negative"
            freq[j] = freq_in[j]
    elif isinstance(freq_in, np.ndarray):
        freq = freq_in.astype(int)
        assert np.all(freq >= 0), "All frequencies must be integer and non-negative"
    # Check divisibility of frequencies
    assert np.all(np.equal(np.mod(np.max(freq) / freq, 1), 0)), "The maximal frequency must be divisible by all other frequencies"
    return freq

def data_to_nparray(data):
    v = None
    if (type(data) is pd.DataFrame) or (type(data) is pd.Series):
        v = data.to_numpy(copy=True)
    elif type(data) is np.ndarray:
        v = np.copy(data)
    if (not v is None) and (v.ndim == 1):
        # Mutate to column vector
        v = np.transpose(np.atleast_2d(v))
    return v

# ----------------------------------------------------------------
# Almon lag functions

def almon(K, theta1, theta2, beta):
    f1 = np.exp(theta1 * np.arange(K))
    f2 = np.exp(theta2 * np.arange(K)**2)
    w  = np.multiply(f1, f2)
    return beta * w / np.sum(w)

def almon_nonorm(K, theta1, theta2, beta):
    f1 = np.exp(theta1 * np.arange(K))
    f2 = np.exp(theta2 * np.arange(K)**2)
    w  = np.multiply(f1, f2)
    return beta * w

def almon_k(k, theta1, theta2):
    return np.exp(theta1*k + theta2*(k**2))

# ----------------------------------------------------------------
# MIDAS Class
#

class MIDAS:
    def __init__(self, freq, ylags, xlags):
        self.freq_  = check_freq(freq)
        self.ylags_ = check_lags(ylags)
        self.xlags_ = check_lags(xlags)

        # Check that, if vectors, 'freq' and 'xlags' have the same length
        assert len(self.freq_) == len(self.xlags_), "Specifications 'freq' and 'xlags' do not have compatible lengths"
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mixed Frequency [FORECASTING] 
    #   Regression Matrices Generator

    def mixed_freq_regmats_for(self, Y, X, steps=0, freq=None, ylags=None, xlags=None, debug=False):
        #   Convert data of different frequencies -- Y being sampled
        #   at lower frequency rate that X -- to regression matrices
        #   for MIDAS regression
        #
        #   Low-freq data is repeated and aligned appropriately to 
        #   account for different timings

        X_TUPLE_FLAG = isinstance(X, tuple)

        # Flatten data
        # NOTE: avoid this here, assume it has already been done in higher
        #       level function call

        # Check sample size(s) of regressors versus response
        Ty, Ky = Y.shape
        Tx = np.array([X_j.shape[0] for X_j in X]) if X_TUPLE_FLAG else X.shape[0]
        Kx = len(X) if X_TUPLE_FLAG else X.shape[1]

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        assert np.all(Tx % Ty == 0), "Sample size of (series in) 'X' not a multiple of sample size of 'Y'"

        if debug:
            print("---")
            print("MIDAS.mixed_freq_regmats_for() -> Debug Info")
            print(f" Y           ->  {Ky} variable(s)\t [low-frequency]")
            print(f" X           ->  {Kx} variable(s)\t [high-frequency]")

        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_

        if len(ylags_) > 1:
            raise ValueError("Length of 'ylags' is greater than 1")

        if len(xlags_) == 1:
            xlags_ = np.repeat(xlags_, Kx)
        else:
            assert (len(xlags_) == Kx), "Length of 'xlags' does not coincide with total number of Y covariates"

        # NOTE: if all regressors are at the same (high) frequency,
        # then repeat 'freq_' by the length of 'Kx'

        if (not X_TUPLE_FLAG) and (len(freq_) == 1):
            freq_ = np.repeat(freq_, Kx)
        
        # We work at the highest frequency, therefore all
        # regression matrices for low-frequency data are augmented by
        # repeating data. We assume that *for each frequency* data 
        # is released at the *last possible time index* of the period

        Tmax = np.max(Tx)
        Fmax = np.max(freq_)
        assert floor(Tmax / Fmax) == Ty

        # Check: compare 'Ty' with individual 'X' series lengths and
        # warn user in case these do not coincide with 'freq' values
        # (not necessarily an error)
        if not np.all(Tx / Ty == freq):
            print("---")
            print("WARNING: frequencies of regressors might be incorrect")
            with np.printoptions(precision=2, suppress=True):
                print(f"Tx / Ty   =  {Tx / Ty}")
                print(f"freq      =  {freq}")

        # To build regression matrices, proceed this way:
        #   (1) Create pre-matrices that are of time size
        #         Tmax
        #
        #   (2) Cut pre-matrices according to largest feasible
        #       regression sample by 'ylags' and 'xlags'

        # Frequency multiplier
        Fm = np.floor(Fmax / freq_).astype(int)

        #h1   = np.ceil((xlags_+1)/frequency_)
        #h2   = np.ceil(steps/frequency_) + 1
        #tau1 = np.max(np.hstack((ylags_, h1, h2))).astype(int)

        h1   = (xlags_+1) / freq_ 
        h2   = np.array((steps / Fmax,)) 
        #tau1 = np.max(np.concatenate((ylags_, np.floor(h1 + h2)))).astype(int)
        tau1 = np.max(np.concatenate((ylags_, np.ceil(h1), np.ceil(h2)))).astype(int)

        if debug:
            print("---")
            print(f" h1    =  {h1}")
            print(f" h2    =  {h2}")
            print(f" tau1  =  {tau1}")
            print("---")

        # (1)
        # Y response pre-matrix
        pre_Yresp_for = np.full((Tmax, 1), np.nan)
        if not ylags_ == 0:
            pre_Yresp_for[Fmax:(Tmax-Fmax),]  = np.repeat(Y[2:Ty,], Fmax, axis=0)
            pre_Yresp_for[0:(Fmax*(tau1-1)),] = np.nan
        else:
            pre_Yresp_for[0:(Tmax-Fmax),] = np.repeat(Y[1:Ty,], Fmax, axis=0)
        
        # Y lagged covariates pre-matrix
        pre_Yregs_for = np.full((Tmax, 1), np.nan)
        pre_Yregs_for[(Fmax):(Tmax-1),] = np.repeat(Y[0:Ty-1,], Fmax, axis=0)[1:]
        pre_Yregs_for[-1,] = Y[-1,]

        # X lagged covariates pre-matrix
        pre_Xregs_for = np.full((Tmax, Kx), np.nan)
        if X_TUPLE_FLAG:
            for k in range(Kx):
                X_k_rep = np.repeat(np.squeeze(X[k]), floor(Fmax / freq[k]), axis=0)
                pre_Xregs_for[(Fm[k]-1):,k] = X_k_rep[0:(Tmax-Fm[k]+1)]
        else:
            pre_Xregs_for = X
                
        #print("pre_Yresp_now:")
        #print(pre_Yresp_now)
        #print("pre_Yregs_now:")
        #print(pre_Yregs_now)
        #print("pre_Xregs_now:")
        #print(pre_Xregs_now)

        # (2)
        # Y response matrix
        Yresp_for = pre_Yresp_for

        # Y regression matrix
        if (ylags_ == 0):
            Yregs_for = None
        else:
            Yregs_for = np.full((Tmax, np.sum(ylags_)), np.nan)
            # Stack lags
            m = 0  
            for k in range(Ky):
                ylags_k = ylags_[k]
                for j in range(ylags_k):
                    Yregs_for[(j*Fmax):Tmax, m] = pre_Yregs_for[0:(Tmax-j*Fmax), k]
                    m += 1
        
        # X regression matrix
        # NOTE: 'xlags == 0' means that the MIDAS regression matrix will still
        #       contain at least the 'contemporaneous' X regressors, thus
        #       increase 'xlags' by 1
        Xregs_for = np.full((Tmax, np.sum((xlags_+1))), np.nan)
        # Stack lags
        m = 0
        for k in range(Kx):
            xlags_k = xlags_[k]
            for j in range(xlags_k+1):
                Xregs_for[(j*Fm[k]):Tmax, m] = pre_Xregs_for[0:(Tmax-j*Fm[k]), k]
                m += 1

        if debug:
            print(" Forecasting regression matrices shapes:")
            if not Yregs_for is None:
                print(f"  {Yresp_for.shape}   ~  [AR] {Yregs_for.shape}, [weighted] {Xregs_for.shape}")
            else:
                print(f"  {Yresp_for.shape}   ~  [weighted] {Xregs_for.shape}")
            print(" Preliminary joint regression matrix:")
            if not Yregs_for is None:
                print(np.hstack((Yresp_for, Yregs_for, Xregs_for))[:36])
            else:
                print(np.hstack((Yresp_for, Xregs_for))[:36])

        return (Yresp_for, Yregs_for, Xregs_for)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MIDAS Fitting [FORECASTING] 

    def fitAlmon_prep_data_for(self, Y, X, steps, freq, ylags, xlags, debug=False):
        # [Sub-routine] Process data to create regression matrices

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # Check data types and flatten
        Y_ = data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of 'Y' not recognized, need pandas.DataFrame or numpy.ndarray.")
        
        if isinstance(X, tuple) or isinstance(X, list):
            X_pre = tuple(X)
            # Check that 'X' tuple size is equal to the length of 'freq'
            assert len(X_pre) == len(freq), "Number of regressors in 'X' and frequencies in 'freq' are different"
 
            X_ = []
            for j in range(len(X_pre)):
                # Since 'X' is a tuple of data, check that each individual
                # element contains only 1 series
                assert X_pre[j].shape[1] == 1, "All elements in tuple 'X' must be single-column series"

                X_j = data_to_nparray(X_pre[j])
                if X_ is None:
                    raise TypeError(f"Type of 'X' series at index {j} not recognized, need pandas.DataFrame or numpy.ndarray.")
                else:
                    X_.append(X_j)
            X_ = tuple(X_)
        else:
            X_ = data_to_nparray(X)
            if X_ is None:
                    raise TypeError("Type of 'X' not recognized, need pandas.DataFrame or numpy.ndarray.")

            # Check that 'freq' is a just a single frequency
            assert len(freq) == 1

        # Check sample size(s) of regressors versus response
        Ty, Ky = Y_.shape
        Tx = np.array([X_j.shape[0] for X_j in X_]) if isinstance(X_, tuple) else X_.shape[0]
        Kx = len(X_) if isinstance(X_, tuple) else X_.shape[1]

        assert np.all(Tx % Ty == 0), "Sample size of (series in) 'X' not a multiple of sample size of 'Y'"

        # Get regression matrices 
        Yresp_for, Yregs_for, Xregs_for = self.mixed_freq_regmats_for(Y=Y_, X=X_, steps=steps, freq=freq, ylags=ylags, xlags=xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # Extract subset with full observations
        # NOTE: 'Zt' will be the full regression matrix, including intercept
        if not dynamic:
            maskY1 = np.minimum.reduce(np.bitwise_not(np.isnan(Yresp_for)), axis=1)
            #maskX  = np.minimum.reduce(np.bitwise_not(np.isnan(Xregs_for)), axis=1)
            mask   = maskY1 #* maskX
            T      = np.sum(mask)
            Yt_for = Yresp_for[mask,]
            Zt_for = np.hstack((np.ones((T, 1)), Xregs_for[mask,]))
        else:
            maskY1 = np.minimum.reduce(np.bitwise_not(np.isnan(Yresp_for)), axis=1)
            maskY0 = np.minimum.reduce(np.bitwise_not(np.isnan(Yregs_for)), axis=1)
            #maskX  = np.minimum.reduce(np.bitwise_not(np.isnan(Xregs_for)), axis=1)
            mask   = maskY1 * maskY0 #* maskX
            T      = np.sum(mask)
            Yt_for = Yresp_for[mask,]
            Zt_for = np.hstack((np.ones((T, 1)), Yregs_for[mask,], Xregs_for[mask,]))

        # Check that the mask and data have the corret shape
        assert T % np.max(freq) == 0

        if debug:
            print("---")
            print("MIDAS.fitAlmon_prep_data_for() -> Debug Info")
            print("Compacted joint regression matrix:")
            print(np.hstack((Yt_for, Zt_for))[:36])
                
        return (Yt_for, Zt_for, mask, T, Ky, Kx)

    def fitAlmon_for(self, Y, X, steps=1, method='L-BFGS-B', theta0=None, normalize=False, grad=True, debug=False):
        # Fit a MIDAS regression model with exponential Almon lags

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # Get self parameters
        ylags = self.ylags_
        xlags = self.xlags_
        freq  = self.freq_

        # (1) Create MIDAS regression matrices ---
        Yt_for, Zt_for, mask, T, Ky, Kx = self.fitAlmon_prep_data_for(Y=Y, X=X, steps=steps, freq=freq, ylags=ylags, xlags=xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        # (2) Optimize Almon lag (+ autoregressive) parameters ---
        if grad:
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_grad(Yt_for, Zt_for, steps, Ky, Kx, freq, ylags, xlags, method=method, theta0=theta0, normalize=normalize, debug=debug)
        else:
            if not theta0 is None:
                print("Initial MIDAS parameter value 'thet0' ignored")
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_for(Yt_for, Zt_for, steps, Ky, Kx, freq, ylags, xlags, method=method, normalize=normalize, debug=debug)

        # Output
        Yt_fit_for    = []
        Residuals_for = []
        RSS_for       = []
        for s in range(steps):
            Yt_fit_for.append(parsfit_midas_for[s]['Yt_fit'])
            Residuals_for.append(parsfit_midas_for[s]['Residuals'])
            RSS_for.append(parsfit_midas_for[s]['RSS'])

        fit_out = {
            'parsfit_for':  parsfit_midas_for,
            'Y_fit':        Yt_fit_for,
            'Residuals':    Residuals_for,
            'RSS':          RSS_for,
            'steps':        steps,
            'T':            T,
            'Y':            Yt_for,
            'Z':            Zt_for,
            'z_slices':     for_zslices,
            'y_slices':     for_yslices,
            'freq':         freq,
            'ylags':        ylags,
            'xlags':        xlags,
            'Ky':           Ky,
            'Kx':           Kx,
            'mask':         mask,
            'method':       method,
            'dynamic':      dynamic,
            'normalize':    normalize,
            'type':         'Almon',
            'grad':         grad,
        }

        return fit_out

    def fitAlmon_solve_for(self, Yt, Zt, steps, Ky, Kx, freq, ylags, xlags, method='nls', normalize=False, debug=False):
        # [Sub-routine] Solve for the optimal MIDAS parameters
        # Slice the data into subsets before optimization: each 
        # subset indexes a specific window for data release following 
        # the high-frequency variables.
        # Since there are exactly R = max('freq') release windows, 
        # fit a number max('freq') of distinct models.

        if debug:
            print("---")
            print("MIDAS.fitAlmon_solve_for() -> Debug Info")
            print(f"Optimization method: {method}")

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_
        
        if len(ylags_) > 1:
            raise ValueError("Length of 'ylags' is greater than 1")

        if len(xlags_) == 1:
            xlags_ = np.repeat(xlags_, Kx)
        else:
            assert (len(xlags_) == Kx), "Length of 'xlags' does not coincide with total number of Y covariates"

        # Highest frequency
        Fmax = np.max(freq_)
        
        # Sample slicing indexes
        #h1   = (xlags_+1)/freq
        #h2   = steps/Fmax
        #tau1 = np.max(np.hstack((ylags_, np.floor(h1 + h2)))).astype(int)

        tau2 = np.sum(np.maximum.reduce(np.isnan(Zt), axis=1))

        if debug:
            print("---")
            print(f" tau2  =  {tau2}")
            print("---")
        
        # Generate slicing indexes 
        T = len(Yt)
        for_zslices = []
        for_yslices = []
        j = (Fmax-1)
        for s in range(steps):
            r = j - (s % Fmax)
            q = ceil((s+1) / Fmax)
            # NOTE: handle cases where 'NaN' observations from high-freq.
            #       lags must be excluded in the first block
            if r < tau2:
                r = Fmax + r
                q = q + 1
            # 
            for_zslices.append(np.arange(r, T-s, Fmax))
            for_yslices.append(np.arange(q*Fmax - 1, T, Fmax))

        if debug:
            print("Regressors slicing indexes [for_zslices]:")
            for s in range(steps):
                print(f"s = {s+1}: {for_zslices[s]}")
            print("Response slicing indexes [for_yslices]:")
            for s in range(steps):
                print(f"s = {s+1}: {for_yslices[s]}")
        
        # Define weight function
        if normalize:
            def phi_MIDAS(theta):
                phi = np.zeros(1+np.sum(ylags_)+np.sum(xlags_+1))
                p = 1+np.sum(ylags_)
                phi[0:p] = theta[0:p]
                # Almond lag parameters
                m = p
                for k in range(Kx):
                    q_k = xlags_[k]+1
                    phi_k = almon(q_k, theta[p+3*k], theta[p+3*k+1], theta[p+3*k+2])
                    phi[m:(m+q_k)] = phi_k
                    m += q_k
                return phi
        
        else:
            def phi_MIDAS(theta):
                phi = np.zeros(1+np.sum(ylags_)+np.sum(xlags_+1))
                p = 1+np.sum(ylags_)
                phi[0:p] = theta[0:p]
                # Almond lag parameters
                m = p
                for k in range(Kx):
                    q_k = xlags_[k]+1
                    phi_k = almon_nonorm(q_k, theta[p+3*k], theta[p+3*k+1], theta[p+3*k+2])
                    phi[m:(m+q_k)] = phi_k
                    m += q_k
                return phi

        # Init parameter vector
        theta0 = np.zeros(1+np.sum(ylags_)+(Kx*3))
        #theta0     = 0.001 * np.ones(1+np.sum(ylags_)+(Kx*3))

        # Optimize
        if method == 'nls':
            # Define objective function
            def Res_obj(theta, i):
                return np.squeeze(Yt[for_yslices[i],:] - Zt[for_zslices[i],:] @ np.vstack(phi_MIDAS(theta)))

            # Solve using 'scipy.optim.least_squares' (nonlinear least-squares)
            res = []
            for s in range(steps):
                verbose = 2 if debug else 1
                res_s = optim.least_squares(lambda _theta_ : Res_obj(_theta_, s), theta0, 
                            max_nfev=200000, ftol=1e-8, xtol=1e-8, gtol=1e-8, method='lm', verbose=verbose)
                
                res.append({
                    'x':            res_s.x,
                    'fun':          np.sum(res_s.fun**2),
                    'status':       res_s.status,
                    'message':      res_s.message,
                })

                if debug:
                    print("---")
                    print(f"s = {s+1} / {steps} | [h = {Fmax+s} / {steps+Fmax-1}]")
                    print("Joint regression matrix:")
                    with np.printoptions(precision=4, suppress=True):
                        print(np.hstack((Yt[for_yslices[s],:], Zt[for_zslices[s],:])))
                    print("NLS optimizer [theta_opt]:")
                    print(res_s.x)

        #elif method == 'nelmead':
        #    def RSS_obj(theta, i):
        #        c = floor(i / frequency) * frequency
        #        return np.sum(np.square(np.squeeze(Yt[c:][for_yslices[i],Yidx_:] - Zt[:(T-c)][for_zslices[i],:] @ np.vstack(phi_MIDAS(theta)))))
    
        #    res = optim.minimize(RSS_obj, theta0, method='nelder-mead', 
        #                  options={'maxiter': 1e4, 'disp': debug, 'adaptive': True})
        
        else:
            raise ValueError("Incorrect optimization method selected") 

        parsfit_midas_for = []
        # NOTE: convention is that 0 < h < max(freq) indexes *nowcasting* horizons,
        #       therefore index models starting from h = max(freq)
        for s in range(steps):
            # Outputs
            theta_opt_s = res[s]['x']
            phi_opt_s   = phi_MIDAS(theta_opt_s)
            RSS_s       = res[s]['fun']
            status_s    = res[s]['status']
            message_s   = res[s]['message']

            # NOTE: COMPARISON WITH MATLAB (manual debug only)
            #theta1      = 0.03 * np.ones(1+np.sum(ylags_)+(Kx*3))
            #theta_opt_s = theta1
            #phi_opt_s   = phi_MIDAS(theta1)

            Yt_fit_s    = Zt[for_zslices[s],:] @ np.vstack(phi_opt_s)
            Residuals_s = Yt[for_yslices[s],:] - Yt_fit_s

            parsfit_midas_for.append({
                's':                s+1,
                'h':                s+Fmax,
                'theta_opt':        theta_opt_s,
                'phi_opt':          phi_opt_s,
                'theta0':           theta0,
                'Yt_fit':           Yt_fit_s,
                'Residuals':        Residuals_s,
                'RSS':              RSS_s,
                'optim_status':     status_s,
                'optim_message':    message_s,
            })

        return parsfit_midas_for, for_zslices, for_yslices

    def fitAlmon_solve_grad(self, Yt, Zt, steps, Ky, Kx, freq, ylags, xlags, method='L-BFGS-B', theta0=None, normalize=False, debug=False):
        # [Sub-routine] Solve for the optimal MIDAS parameters 
        # This method is an implementation of vectorized loss and gradient
        # for the non-normalized Almon lag weights.

        if debug:
            print("MIDAS.fitAlmon_solve_grad() -> Debug Info")
            print(f"Optimization method: {method}")

        # Notify that this method only supports un-normalized Almon weights
        if normalize:
            print("MIDAS.fitAlmon_solve_grad() [!] Explicit gradient only uses non-normalized Almon weights")

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_
        
        if len(ylags_) > 1:
            raise ValueError("Length of 'ylags' is greater than 1")

        if len(xlags_) == 1:
            xlags_ = np.repeat(xlags_, Kx)
        else:
            assert (len(xlags_) == Kx), "Length of 'xlags' does not coincide with total number of Y covariates"

        # Highest frequency
        Fmax = np.max(freq_)

        # Sample slicing indexes
        #h1   = (xlags_+1)/freq
        #h2   = steps/Fmax
        #tau1 = np.max(np.hstack((ylags_, np.floor(h1 + h2)))).astype(int)

        tau2 = np.sum(np.maximum.reduce(np.isnan(Zt), axis=1))

        if debug:
            print("---")
            print(f" tau2  =  {tau2}")
            print("---")
        
        # Generate slicing indexes 
        T = len(Yt)
        for_zslices = []
        for_yslices = []
        j = (Fmax-1)
        for s in range(steps):
            r = j - (s % Fmax)
            q = ceil((s+1) / Fmax)
            # NOTE: handle cases where 'NaN' observations from high-freq.
            #       lags must be excluded in the first block
            if r < tau2:
                r = Fmax + r
                q = q + 1
            # 
            for_zslices.append(np.arange(r, T-s, Fmax))
            for_yslices.append(np.arange(q*Fmax - 1, T, Fmax))

        if debug:
            print("Regressors slicing indexes [for_zslices]:")
            for s in range(steps):
                print(f"s = {s+1}: {for_zslices[s]}")
            print("Response slicing indexes [for_yslices]:")
            for s in range(steps):
                print(f"s = {s+1}: {for_yslices[s]}")

        # Pre-computed for faster evaluation
        Nu   = np.vstack(np.concatenate([np.arange(1+xl) for xl in xlags_]))
        Nusq = Nu**2
        B    = np.zeros((np.sum(xlags_+1), Kx)) # np.kron(np.eye(Kx), np.ones((1+xlags, 1)))
        m = 0
        for k in range(Kx):
            B[m:(m+(xlags_[k]+1)),k] = np.ones(1+xlags_[k])
            m += xlags_[k]+1

        # Sum-of-squares 
        def RSS_obj(pars, i):
            # Parameters
            p = 1+np.sum(ylags_) #(ylags*Ky)
            rho    = np.vstack(pars[0:p])
            beta   = np.vstack(pars[p:(p+Kx)])
            theta1 = np.vstack(pars[(p+Kx):(p+Kx*2)])
            theta2 = np.vstack(pars[(p+Kx*2):])
            bephi  = (B @ beta) * np.exp((B @ theta1) * Nu + (B @ theta2) * Nusq) 
            return np.sum(np.square(Yt[for_yslices[i],:] - Zt[for_zslices[i],:] @ np.vstack((rho, bephi))))

        # Gradient
        def RSS_obj_grad(pars, i):
            # Parameters
            p = 1+np.sum(ylags_) #(ylags*Ky)
            rho    = np.vstack(pars[0:p])
            beta   = np.vstack(pars[p:(p+Kx)])
            theta1 = np.vstack(pars[(p+Kx):(p+Kx*2)])
            theta2 = np.vstack(pars[(p+Kx*2):])
            phi    = np.exp((B @ theta1) * Nu + (B @ theta2) * Nusq) 
            c = Yt[for_yslices[i],:] - Zt[for_zslices[i],:] @ np.vstack((rho, (B @ beta) * phi))
            # Components
            G1 = (Zt[for_zslices[i],0:p]).T @ c
            D  = (Zt[for_zslices[i],p:]).T @ c
            G2 = B.T @ np.diagflat(phi) @ D
            G3 = np.diagflat(B @ beta) @ D
            H1 = B.T @ np.diagflat(phi * Nu)
            H2 = B.T @ np.diagflat(phi * Nusq)
            return np.squeeze(- 2 * np.vstack((G1, G2, H1 @ G3, H2 @ G3)))

        # Optimized theta to phi parameters
        def phi_MIDAS(pars_opt):
            p = 1+np.sum(ylags_)
            #q = np.sum(xlags_+1)
            beta   = pars_opt[p:(p+Kx)]
            theta1 = pars_opt[(p+Kx):(p+Kx*2)]
            theta2 = pars_opt[(p+Kx*2):]
            # Organize optimized MIDAS parameters 
            phi_opt = np.zeros(1+np.sum(ylags_)+np.sum(xlags_+1))
            phi_opt[0:p] = pars_opt[0:p]
            m = p
            for k in range(Kx):
                phi_k = almon_nonorm(xlags_[k]+1, theta1[k], theta2[k], beta[k])
                phi_opt[m:(m+(xlags_[k]+1))] = phi_k
                m += xlags_[k]+1
            return phi_opt


        # Init parameter vector
        #theta0 = np.concatenate((np.ones(1+(ylags*Ky)+Kx), np.zeros(Kx*2)))
        #theta0 = 1e-2 * np.ones(1+np.sum(ylags_)+(Kx*3))
        #theta0 = 1e-6 * np.ones(1+np.sum(ylags_)+(Kx*3))

        theta0 = np.zeros(1+np.sum(ylags_)+(Kx*3)) if (theta0 is None) else theta0

        #RSS_obj(theta0, 0)
        #RSS_obj_grad(theta0, 0)

        # Optimize
        res = []
        for s in range(steps): 
            RSS_obj_s       = lambda _theta_ : RSS_obj(_theta_, s)
            RSS_obj_grad_s  = lambda _theta_ : RSS_obj_grad(_theta_, s)
            
            res_s = None

            if method == 'BFGS':
                res_s = optim.minimize(fun=RSS_obj_s, x0=theta0, jac=RSS_obj_grad_s, method='BFGS', 
                                options={'maxiter': 1e9, 'disp': debug})

            elif method == 'L-BFGS-B':
                disp = 50 if debug else None
                res_s = optim.minimize(fun=RSS_obj_s, x0=theta0, jac=RSS_obj_grad_s, method='L-BFGS-B', 
                                options={'maxiter': 1e9, 'disp': disp})

            elif method == 'SLSQP':
                res_s = optim.minimize(fun=RSS_obj_s, x0=theta0, jac=RSS_obj_grad_s, method='SLSQP', 
                                options={'maxiter': 1e9, 'disp': debug})

            else:
                raise ValueError("Incorrect optimization method selected") 

            res.append({
                    'x':            res_s.x,
                    'fun':          res_s.fun,
                    'status':       res_s.status,
                    'message':      res_s.message,
                })

            if debug:
                print("---")
                print(f"s = {s+1} / {steps} | [h = {Fmax+s} / {steps+Fmax-1}]")
                print("Joint regression matrix:")
                with np.printoptions(precision=4, suppress=True):
                    print(np.hstack((Yt[for_yslices[s],:], Zt[for_zslices[s],:])))
                print(f"{method} optimizer [theta_opt]:")
                print(res_s.x)


        parsfit_midas_for = []
        # NOTE: convention is that 0 < h < max(freq) indexes *nowcasting* horizons,
        #       therefore index models starting from h = max(freq)
        for s in range(steps):
            # Outputs
            theta_opt_s = res[s]['x']
            phi_opt_s   = phi_MIDAS(theta_opt_s)
            RSS_s       = res[s]['fun']
            status_s    = res[s]['status']
            message_s   = res[s]['message']

            # NOTE: COMPARISON WITH MATLAB (manual debug only)
            #theta1      = 0.03 * np.ones(1+np.sum(ylags_)+(Kx*3))
            #theta_opt_s = theta1
            #phi_opt_s   = phi_MIDAS(theta1)

            Yt_fit_s    = Zt[for_zslices[s],:] @ np.vstack(phi_opt_s)
            Residuals_s = Yt[for_yslices[s],:] - Yt_fit_s

            parsfit_midas_for.append({
                's':                s+1,
                'h':                s+Fmax,
                'theta_opt':        theta_opt_s,
                'phi_opt':          phi_opt_s,
                'theta0':           theta0,
                'Yt_fit':           Yt_fit_s,
                'Residuals':        Residuals_s,
                'RSS':              RSS_s,
                'optim_status':     status_s,
                'optim_message':    message_s,
            })

        return parsfit_midas_for, for_zslices, for_yslices

    def fitAlmon_getRSS_mats(self, Y, X, fit_length=None, steps=1, freq=None, ylags=None, xlags=None, debug=False):
        # Prepare matrices and slices to allow quick numerical evaluation
        # of the RSS objective and its gradient via the 'fitAlmon_RSS()'
        # function.

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_

        # Get MIDAS regression matrices
        Yt_full, Zt_full, _, T, Ky, Kx = self.fitAlmon_prep_data_for(Y=Y, X=X, steps=steps, freq=freq_, ylags=ylags_, xlags=xlags_, debug=debug)

        # NOTE: bare minimum checks, so might break if inputs are improper

        # Highest frequency
        Fmax = np.max(freq_)

        h1   = (xlags_+1)/freq_
        h2   = np.array((steps/Fmax,)) 
        tau1 = np.max(np.concatenate((ylags_, np.ceil(h1), np.ceil(h2)))).astype(int)
        
        # NOTE: without lags of 'Y', 'Tcutoff' can be directly
        #       simplified to 'fit_length * Fmax'
        if ylags_ == 0:
            Tcutoff = fit_length * Fmax
        else:
            Tcutoff = (fit_length - tau1) * Fmax

        assert (T > Tcutoff), "Sample size T is too small compared to training length"

        # Fitting data
        Yt_for = Yt_full[0:Tcutoff,]
        Zt_for = Zt_full[0:Tcutoff,]
        T = Tcutoff

        # Sample slicing indexes
        tau2 = np.sum(np.maximum.reduce(np.isnan(Zt_for), axis=1))
        
        # Generate slicing indexes 
        #T = len(Yt_for)
        for_zslices = []
        for_yslices = []
        j = (Fmax-1)
        for s in range(steps):
            r = j - (s % Fmax)
            q = ceil((s+1) / Fmax)
            # NOTE: handle cases where 'NaN' observations from high-freq.
            #       lags must be excluded in the first block
            if r < tau2:
                r = Fmax + r
                q = q + 1
            # 
            for_zslices.append(np.arange(r, T-s, Fmax))
            for_yslices.append(np.arange(q*Fmax - 1, T, Fmax))

        # Pre-computed for faster evaluation
        Nu   = np.vstack(np.concatenate([np.arange(1+xl) for xl in xlags_]))
        Nusq = Nu**2
        B    = np.zeros((np.sum(xlags_+1), Kx)) # np.kron(np.eye(Kx), np.ones((1+xlags, 1)))
        m = 0
        for k in range(Kx):
            B[m:(m+(xlags_[k]+1)),k] = np.ones(1+xlags_[k])
            m += xlags_[k]+1

        return Yt_for, Zt_for, steps, ylags_, xlags_, Ky, Kx, for_yslices, for_zslices, Nu, Nusq, B

    def fitAlmon_RSS_obj(self, datamats, pars, i=0):
        # Compute RSS
        Yt, Zt, steps, ylags_, _, _, Kx, for_yslices, for_zslices, Nu, Nusq, B = datamats

        # Check bound for i
        assert ((i >= 0) and (i < steps)), "Index 'i' is out of bounds"

        # Residual Sum-of-squares 
        p = 1+np.sum(ylags_) 
        rho    = np.vstack(pars[0:p])
        beta   = np.vstack(pars[p:(p+Kx)])
        theta1 = np.vstack(pars[(p+Kx):(p+Kx*2)])
        theta2 = np.vstack(pars[(p+Kx*2):])
        bephi  = (B @ beta) * np.exp((B @ theta1) * Nu + (B @ theta2) * Nusq) 
        rss = np.sum(np.square(Yt[for_yslices[i],:] - Zt[for_zslices[i],:] @ np.vstack((rho, bephi))))
        
        return rss

    def fitAlmon_RSS_grad(self, datamats, pars, i=0):
         # Compute RSS
        Yt, Zt, steps, ylags_, _, _, Kx, for_yslices, for_zslices, Nu, Nusq, B = datamats

        # Check bound for i
        assert ((i >= 0) and (i < steps)), "Index 'i' is out of bounds"

        # RSS Gradient
        p = 1+np.sum(ylags_) #(ylags*Ky)
        rho    = np.vstack(pars[0:p])
        beta   = np.vstack(pars[p:(p+Kx)])
        theta1 = np.vstack(pars[(p+Kx):(p+Kx*2)])
        theta2 = np.vstack(pars[(p+Kx*2):])
        phi    = np.exp((B @ theta1) * Nu + (B @ theta2) * Nusq) 
        c = Yt[for_yslices[i],:] - Zt[for_zslices[i],:] @ np.vstack((rho, (B @ beta) * phi))
        # 
        G1 = (Zt[for_zslices[i],0:p]).T @ c
        D  = (Zt[for_zslices[i],p:]).T @ c
        G2 = B.T @ np.diagflat(phi) @ D
        G3 = np.diagflat(B @ beta) @ D
        H1 = B.T @ np.diagflat(phi * Nu)
        H2 = B.T @ np.diagflat(phi * Nusq)
        rss_grad = np.squeeze(- 2 * np.vstack((G1, G2, H1 @ G3, H2 @ G3)))
        
        return rss_grad

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MIDAS Fixed-Parameter Forecasting [FORECASTING]

    def fixedparamsForecast(self, data, fit_length=None, steps=1, method='nls', normalize=True, grad=False, theta0=None, debug=False):
        # Forecast MIDAS model "in sample" i.e. to evaluate model
        # forecasting performance

        # Unpack data, options
        Y, X = data

        if debug:
            print("---")
            print("MIDAS.fixedparamsForecast() | MIDAS model [Forecasting] -> Debug Info")

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # Get self parameters, makes expressions easier to read
        ylags = self.ylags_
        xlags = self.xlags_
        freq  = self.freq_

        # (1) Create MIDAS regression matrices ---
        Yt_full, Zt_full, mask, Tfull, Ky, Kx = self.fitAlmon_prep_data_for(Y=Y, X=X, steps=steps, freq=freq, ylags=ylags, xlags=xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # Highest frequency
        Fmax = np.max(freq)
        
        # (2) Subset regression matrices based on 'fit_length' 
        #h1   = (xlags+1)/Fmax
        #h2   = steps/Fmax
        #tau1 = np.max(np.hstack((ylags, np.floor(h1 + h2)))).astype(int)
        #Tcutoff = (fit_length-2)*Fmax

        h1   = (xlags+1)/freq
        h2   = np.array((steps/Fmax,)) 
        tau1 = np.max(np.concatenate((ylags, np.ceil(h1), np.ceil(h2)))).astype(int)
        
        # NOTE: without lags of 'Y', 'Tcutoff' can be directly
        #       simplified to 'fit_length * Fmax'
        if ylags == 0:
            Tcutoff = fit_length * Fmax
        else:
            Tcutoff = (fit_length - tau1) * Fmax

        assert (Tfull > Tcutoff), "Sample size T is too small compared to training length"

        # Fitting data
        Yt_forfit = Yt_full[0:Tcutoff,]
        Zt_forfit = Zt_full[0:Tcutoff,]

        # Test data
        Yt_fortest = Yt_full[Tcutoff:,]
        # For regressors, must include more data from 'Zt_full' depending
        # on number of steps and lags as given by 'tau1'
        Zt_fortest = Zt_full[(Tcutoff-(tau1-1)*Fmax):,]
        
        # (3) Optimize Almon lag (+ autoregressive) parameters ---
        if grad:
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_grad(Yt_forfit, Zt_forfit, steps, Ky, Kx, freq, ylags, xlags, method=method, theta0=theta0, normalize=normalize, debug=debug)
        else:
            if not theta0 is None:
                print("Initial MIDAS parameter value 'theta0' ignored")
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_for(Yt_forfit, Zt_forfit, steps, Ky, Kx, freq, ylags, xlags, method=method, normalize=normalize, debug=debug)

        # (4) Compute forecasts 
        yslice = np.arange(0, len(Yt_fortest), Fmax)
        
        forecast = []
        for s in range(steps):
            r = (tau1*Fmax-1) - s
            zslice = np.arange(r, len(Zt_fortest)-s, Fmax)

            # Compute forecast
            phi_s = parsfit_midas_for[s]['phi_opt']
            Yt_fortest_s    = Zt_fortest[zslice,:] @ np.vstack(phi_s)
            Error_fortest_s = Yt_fortest[yslice,:] - Yt_fortest_s
            FESS_s          = np.sum(Error_fortest_s**2)

            if debug:
                print("---")
                print(f"s = {s+1} | [h = {Fmax+s} / {steps+Fmax-1}]")
                print("Regressors slicing indexes [zslice]:")
                print(zslice)
                print("Response slicing indexes [yslice]:")
                print(yslice)
                print("Joint forecast matrix:")
                #c = floor(s / frequency) * frequency
                with np.printoptions(precision=4, suppress=True):
                    print(np.hstack((Yt_fortest[yslice,:], Zt_fortest[zslice,:])))

            #
            forecast.append({
                's':            s+1,
                'h':            s+Fmax,
                'Forecast':     Yt_fortest_s,
                'Errors':       Error_fortest_s,
                'FESS':         FESS_s,
                'z_slice':      zslice,
                'Yt_test':      Yt_fortest[yslice,:],
                'Zt_test':      Zt_fortest[zslice,:],
            })

        # End forecasting loop

        # Output
        forecast_out = {
            'parsfit_for':  parsfit_midas_for,
            'forecasts':    forecast,
            'steps':        steps,
            'T':            Tfull,
            'Y':            Yt_full,
            'Z':            Zt_full,
            'z_slices':     for_zslices,
            'y_slices':     for_yslices,
            'freq':         freq,
            'ylags':        ylags,
            'xlags':        xlags,
            'Ky':           Ky,
            'Kx':           Kx,
            'mask':         mask,
            'method':       method,
            'dynamic':      dynamic,
            'normalize':    normalize,
            'type':         'Almon',
            'grad':         grad,
        }

        return forecast_out

    def fixedparamsHighFreqForecast(self, data, fit_length=None, steps=1, method='nls', normalize=True, grad=False, theta0=None, debug=False):
        # Forecast MIDAS model "in sample" i.e. to evaluate model
        # forecasting performance

        # Unpack data, options
        Y, X = data

        if debug:
            print("---")
            print("MIDAS.fixedparamsForecast() | MIDAS model [Forecasting] -> Debug Info")

        # Check forecasting steps 
        assert (type(steps) is int) and (steps > 0), "Forecasting 'steps' must be a positive integer"

        # Get self parameters, makes expressions easier to read
        ylags = self.ylags_
        xlags = self.xlags_
        freq  = self.freq_

        # (1) Create MIDAS regression matrices ---
        Yt_full, Zt_full, mask, Tfull, Ky, Kx = self.fitAlmon_prep_data_for(Y=Y, X=X, steps=steps, freq=freq, ylags=ylags, xlags=xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # Highest frequency
        Fmax = np.max(freq)

        if debug:
            print("---")
            print("! For high-frequency forecast, 'steps' is reference to low-frequency")
            print(f"Number of high-freq steps: {(steps-1)*Fmax + 1}")
        
        # (2) Subset regression matrices based on 'fit_length' 
        #h1   = (xlags+1)/Fmax
        #h2   = steps/Fmax
        #tau1 = np.max(np.hstack((ylags, np.floor(h1 + h2)))).astype(int)
        #Tcutoff = (fit_length-2)*Fmax

        h1   = (xlags+1)/freq
        h2   = np.array((steps/Fmax,)) 
        tau1 = np.max(np.concatenate((ylags, np.ceil(h1), np.ceil(h2)))).astype(int)
        
        # NOTE: without lags of 'Y', 'Tcutoff' can be directly
        #       simplified to 'fit_length * Fmax'
        if ylags == 0:
            Tcutoff = fit_length * Fmax
        else:
            Tcutoff = (fit_length - tau1) * Fmax

        assert (Tfull > Tcutoff), "Sample size T is too small compared to training length"

        # Fitting data
        Yt_forfit = Yt_full[0:Tcutoff,]
        Zt_forfit = Zt_full[0:Tcutoff,]

        # Test data
        Yt_fortest = Yt_full[Tcutoff:,]
        # For regressors, must include more data from 'Zt_full' depending
        # on number of steps and lags as given by 'tau1'
        Zt_fortest = Zt_full[(Tcutoff-(tau1-1)*Fmax):,]
        
        # (3) Optimize Almon lag (+ autoregressive) parameters ---
        fit_steps = (steps-1)*Fmax + 1
        if grad:
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_grad(
                Yt_forfit, Zt_forfit, fit_steps, Ky, Kx, freq, ylags, xlags, method=method, theta0=theta0, normalize=normalize, debug=debug
            )
        else:
            if not theta0 is None:
                print("Initial MIDAS parameter value 'theta0' ignored")
            parsfit_midas_for, for_zslices, for_yslices = self.fitAlmon_solve_for(
                Yt_forfit, Zt_forfit, fit_steps, Ky, Kx, freq, ylags, xlags, method=method, normalize=normalize, debug=debug
            )

        # (4) Compute high-frequency forecasts 
        yslice = np.arange(0, len(Yt_fortest)-Fmax+1)

        forecast_hf = []
        for s in range(steps):
            zslice = np.arange(tau1*Fmax-1, len(Zt_fortest)) - s*Fmax

            # Compute forecast
            phi_s = parsfit_midas_for[s*Fmax]['phi_opt']
            Yt_fortest_s    = Zt_fortest[zslice,:] @ np.vstack(phi_s)
            Error_fortest_s = Yt_fortest[yslice,:] - Yt_fortest_s
            FESS_s          = np.sum(Error_fortest_s**2)

            if debug:
                print("---")
                print(f"s = {s+1}")
                print("Regressors slicing indexes [zslice]:")
                print(zslice)
                print("Response slicing indexes [yslice]:")
                print(yslice)
                print("Joint forecast matrix:")
                #c = floor(s / frequency) * frequency
                with np.printoptions(precision=4, suppress=True):
                    print(np.hstack((Yt_fortest[yslice,:], Zt_fortest[zslice,:])))

            #
            forecast_hf.append({
                's':            s+1,
                'h':            s+Fmax,
                'Forecast':     Yt_fortest_s,
                'Errors':       Error_fortest_s,
                'FESS':         FESS_s,
                'z_slice':      zslice,
                'Yt_test':      Yt_fortest[yslice,:],
                'Zt_test':      Zt_fortest[zslice,:],
            })

        # End forecasting loop

        # Output
        forecast_out = {
            'parsfit_for':  parsfit_midas_for,
            'forecasts':    forecast_hf,
            'steps':        steps,
            'T':            Tfull,
            'Y':            Yt_full,
            'Z':            Zt_full,
            'z_slices':     for_zslices,
            'y_slices':     for_yslices,
            'freq':         freq,
            'ylags':        ylags,
            'xlags':        xlags,
            'Ky':           Ky,
            'Kx':           Kx,
            'mask':         mask,
            'method':       method,
            'dynamic':      dynamic,
            'normalize':    normalize,
            'type':         'Almon',
            'grad':         grad,
        }

        return forecast_out
    
    # #######################################################################################
    # #######################################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mixed Frequency [NOWCASTING] 
    #   Regression Matrices Generator

    def mixed_freq_regmats_now(self, Y, X, freq=None, ylags=None, xlags=None, debug=False):
        #   Convert data of different frequencies -- Y being sampled
        #   at lower frequency rate that X -- to regression matrices
        #   for MIDAS regression 
        #
        #   Low-freq data is repeated and aligned appropriately to 
        #   account for different timings

        X_TUPLE_FLAG = isinstance(X, tuple)

        # Flatten data
        # NOTE: avoid this here, assume it has already been done in higher
        #       level function call

        # Check sample size(s) of regressors versus response
        Ty, Ky = Y.shape
        Tx = np.array([X_j.shape[0] for X_j in X]) if X_TUPLE_FLAG else X.shape[0]
        Kx = len(X) if X_TUPLE_FLAG else X.shape[1]

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        assert np.all(Tx % Ty == 0), "Sample size of (series in) 'X' not a multiple of sample size of 'Y'"

        if debug:
            print("---")
            print("MIDAS.mixed_freq_regmats_now() -> Debug Info")
            print(f" Y           ->  {Ky} variable(s)\t [low-frequency]")
            print(f" X           ->  {Kx} variable(s)\t [high-frequency]")
        
        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_

        if len(ylags_) > 1:
            raise ValueError("Length of 'ylags' is greater than 1")

        if len(xlags_) == 1:
            xlags_ = np.repeat(xlags_, Kx)
        else:
            assert (len(xlags_) == Kx), "Length of 'xlags' does not coincide with total number of Y covariates"

        # NOTE: if all regressors are at the same (high) frequency,
        # then repeat 'freq_' by the length of 'Kx'

        if (not X_TUPLE_FLAG) and (len(freq_) == 1):
            freq_ = np.repeat(freq_, Kx)

        # We work at the highest frequency, therefore all
        # regression matrices for low-frequency data are augmented by
        # repeating data
        #
        # We also assume that for each frequency data is released at the
        # last possible time index of the period

        Tmax = np.max(Tx)
        Fmax = np.max(freq_)
        assert floor(Tmax / Fmax) == Ty

        # Check: compare 'Ty' with individual 'X' series lengths and
        # warn user in case these do not coincide with 'freq' values
        # (not necessarily an error)
        if not np.all(Tx / Ty == freq):
            print("---")
            print("WARNING: frequencies of regressors might be incorrect")
            with np.printoptions(precision=2, suppress=True):
                print(f"Tx / Ty   =  {Tx / Ty}")
                print(f"freq      =  {freq}")

        # To build regression matrices, proceed this way:
        #   (1) Create pre-matrices that are of time size
        #         Tmax
        #
        #   (2) Cut pre-matrices according to largest feasible
        #       regression sample by 'ylags' and 'xlags'

        # Frequency multiplier
        Fm = np.floor(Fmax / freq_).astype(int)

        h1 = (xlags_+1) / freq_
        #tau0 = np.squeeze(np.maximum(ylags_, np.max(np.ceil((xlags_+1)/frequency_))).astype(int))
        tau0 = np.max(np.concatenate((ylags_, np.ceil(h1)))).astype(int)

        if debug:
            print("---")
            print(f" h1    =  {h1}")
            print(f" tau0  =  {tau0}")
            print("---")
         
        # (1)
        # Y response pre-matrix
        pre_Yresp_now = np.full((Tmax, 1), np.nan)
        if not ylags_ == 0:
            pre_Yresp_now[Fmax:,] = np.repeat(Y[1:Ty], Fmax, axis=0)
            pre_Yresp_now[0:(Fmax*(tau0-1)),] = np.nan
        else:
            pre_Yresp_now = np.repeat(Y, Fmax, axis=0)

        # Y lagged covariates pre-matrix
        pre_Yregs_now = np.full((Tmax, 1), np.nan)
        pre_Yregs_now[Fmax:Tmax,] = np.repeat(Y[0:Ty-1,], Fmax, axis=0)

        # X lagged covariates pre-matrix
        pre_Xregs_now = np.full((Tmax, Kx), np.nan)
        if X_TUPLE_FLAG:
            for k in range(Kx):
                X_k_rep = np.repeat(np.squeeze(X[k]), floor(Fmax / freq[k]), axis=0)
                pre_Xregs_now[(Fm[k]-1):,k] = X_k_rep[0:(Tmax-Fm[k]+1)]
        else:
            pre_Xregs_now = X

        #print("pre_Yresp_now:")
        #print(pre_Yresp_now)
        #print("pre_Yregs_now:")
        #print(pre_Yregs_now)
        #print("pre_Xregs_now:")
        #print(pre_Xregs_now)

        # (2)
        # Y response matrix
        Yresp_now = pre_Yresp_now

        # Y regression matrix
        if (ylags_ == 0):
            Yregs_now = None
        else:
            Yregs_now = np.full((Tmax, np.sum(ylags_)), np.nan)
            # Stack lags
            m = 0  
            for k in range(Ky):
                ylags_k = ylags_[k]
                for j in range(ylags_k):
                    Yregs_now[(j*Fmax):Tmax, m] = pre_Yregs_now[0:(Tmax-j*Fmax), k]
                    m += 1
        
        # X regression matrix
        # NOTE: 'xlags == 0' means that the MIDAS regression matrix will still
        #       contain at least the 'contemporaneous' X regressors, thus
        #       increase 'xlags' by 1
        Xregs_now = np.full((Tmax, np.sum((xlags_+1))), np.nan)
        # Stack lags
        m = 0
        for k in range(Kx):
            xlags_k = xlags_[k]
            for j in range(xlags_k+1):
                Xregs_now[(j*Fm[k]):Tmax, m] = pre_Xregs_now[0:(Tmax-j*Fm[k]), k]
                m += 1


        if debug:
            print(" Nowcasting regression matrices shapes:")
            if not Yregs_now is None:
                print(f"  {Yresp_now.shape}   ~  [AR] {Yregs_now.shape}, [weighted] {Xregs_now.shape}")
            else:
                print(f"  {Yresp_now.shape}   ~  [weighted] {Xregs_now.shape}")
            if not Yregs_now is None:
                print(np.hstack((Yresp_now, Yregs_now, Xregs_now))[:36])
            else:
                print(np.hstack((Yresp_now, Xregs_now))[:36])

        return (Yresp_now, Yregs_now, Xregs_now)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MIDAS Fitting [NOWCASTING]

    def fitAlmon_prep_data_now(self, Y, X, freq, ylags, xlags, debug=False):
        # [Sub-routine] Process data to create regression matrices

        # Flatten data
        Y_ = data_to_nparray(Y)
        if Y_ is None:
            raise TypeError("Type of Y not recognized, need pandas.DataFrame or numpy.ndarray.")
        
        if isinstance(X, tuple) or isinstance(X, list):
            X_pre = tuple(X)
            # Check that 'X' tuple size is equal to the length of 'freq'
            assert len(X_pre) == len(freq), "Number of regressors in 'X' and frequencies in 'freq' are different"
                
            X_ = []
            for j in range(len(X_pre)):
                # Since 'X' is a tuple of data, check that each individual
                # element contains only 1 series
                assert X_pre[j].shape[1] == 1, "All elements in tuple 'X' must be single-column series"

                X_j = data_to_nparray(X_pre[j])
                if X_ is None:
                    raise TypeError(f"Type of 'X' series at index {j} not recognized, need pandas.DataFrame or numpy.ndarray.")
                else:
                    X_.append(X_j)
            X_ = tuple(X_)
        else:
            X_ = data_to_nparray(X)
            if X_ is None:
                    raise TypeError("Type of 'X' not recognized, need pandas.DataFrame or numpy.ndarray.")

            # Check that 'freq' is a just a single frequency
            assert len(freq) == 1

        # Check sample size(s) of regressors versus response
        Ty, Ky = Y_.shape
        Tx = np.array([X_j.shape[0] for X_j in X_]) if isinstance(X_, tuple) else X_.shape[0]
        Kx = len(X_) if isinstance(X_, tuple) else X_.shape[1]

        assert np.all(Tx % Ty == 0), "Sample size of (series in) 'X' not a multiple of sample size of 'Y'"

        # Get regression matrices 
        Yresp_now, Yregs_now, Xregs_now = self.mixed_freq_regmats_now(Y=Y_, X=X_, freq=freq, ylags=ylags, xlags=xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # Extract subset with full observations
        # NOTE: 'Zt' will be the full regression matrix, including intercept
        if not dynamic:
            maskY1 = np.minimum.reduce(np.bitwise_not(np.isnan(Yresp_now)), axis=1)
            #maskX  = np.minimum.reduce(np.bitwise_not(np.isnan(Xregs_now)), axis=1)
            mask   = maskY1 #* maskX
            T      = np.sum(mask)
            Yt_now = Yresp_now[mask,]
            Zt_now = np.hstack((np.ones((T, 1)), Xregs_now[mask,]))
        else:
            maskY1 = np.minimum.reduce(np.bitwise_not(np.isnan(Yresp_now)), axis=1)
            maskY0 = np.minimum.reduce(np.bitwise_not(np.isnan(Yregs_now)), axis=1)
            #maskX  = np.minimum.reduce(np.bitwise_not(np.isnan(Xregs_now)), axis=1)
            mask   = maskY1 * maskY0 #* maskX
            T      = np.sum(mask)
            Yt_now = Yresp_now[mask,]
            Zt_now = np.hstack((np.ones((T, 1)), Yregs_now[mask,], Xregs_now[mask,]))

        # Check that the mask and data have the corret shape
        assert T % np.max(freq) == 0

        if debug:
            print("---")
            print("MIDAS.fitAlmon_prep_data_now() -> Debug Info")
            print("Compacted joint regression matrix:")
            print(np.hstack((Yt_now, Zt_now))[:36])

        return (Yt_now, Zt_now, mask, T, Ky, Kx)

    def fitAlmon_now(self, Y, X, method='nls', normalize=True, grad=False, debug=False):
        # Fit a MIDAS regression model with exponential Almon lags

        # Get self parameters, makes expressions easier to read
        ylags = self.ylags_
        xlags = self.xlags_
        freq  = self.freq_

        # (1) Create MIDAS regression matrices ---
        Yt_now, Zt_now, mask, T, Ky, Kx = self.fitAlmon_prep_data_now(Y, X, freq, ylags, xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        # (2) Optimize Almon lag (+ autoregressive) parameters ---
        if grad:
            raise ValueError("TODO")
        else:
            parsfit_midas_now, now_zslices, now_yslices = self.fitAlmon_solve_now(Yt_now, Zt_now, Ky, Kx, freq, ylags, xlags, method=method, normalize=normalize, debug=debug)

        # Output
        Fmax = np.max(freq)

        Yt_fit_now    = []
        Residuals_now = []
        RSS_now       = []
        for h in range(Fmax):
            Yt_fit_now.append(parsfit_midas_now[h]['Yt_fit'])
            Residuals_now.append(parsfit_midas_now[h]['Residuals'])
            RSS_now.append(parsfit_midas_now[h]['RSS'])

        fit_out = {
            'parsfit_now':  parsfit_midas_now,
            'Y_fit':        Yt_fit_now,
            'Residuals':    Residuals_now,
            'RSS':          RSS_now,
            'T':            T,
            'Y':            Yt_now,
            'Z':            Zt_now,
            'z_slices':     now_zslices,
            'y_slices':     now_yslices,
            'freq':         freq,
            'ylags':        ylags,
            'xlags':        xlags,
            'Ky':           Ky,
            'Kx':           Kx,
            'mask':         mask,
            'method':       method,
            'dynamic':      dynamic,
            'normalize':    normalize,
            'type':         'Almon',
            'grad':         grad,
        }

        return fit_out

    def fitAlmon_solve_now(self, Yt, Zt, Ky, Kx, freq, ylags, xlags, method='nls', normalize=False, debug=False):
        # [Sub-routine] Solve for the optimal MIDAS parameters
        # Slice the data into subsets before optimization: each 
        # subset indexes a specific window for data release following 
        # the high-frequency variable.
        # Since there are exactly R=='frequency' release windows, fit as many as 
        # 'frequency' models.

        if debug:
            print("MIDAS.fitAlmon_solve_now() -> Debug Info")
            print(f"Optimization method: {method}")

        # If 'Y' is multivariate, throw error (all regressors must be put in 'X')
        if (Ky > 1):
            raise ValueError("Variable 'Y' is multivariate, please put all relevant regressors in 'X'")

        # If parameters are not explicitly chosen, use the model's own
        freq_  = check_freq(freq) if (not freq is None) else self.freq_
        ylags_ = check_lags(ylags) if (not ylags is None) else self.ylags_
        xlags_ = check_lags(xlags) if (not xlags is None) else self.xlags_
        
        if len(ylags_) > 1:
            raise ValueError("Length of 'ylags' is greater than 1")

        if len(xlags_) == 1:
            xlags_ = np.repeat(xlags_, Kx)
        else:
            assert (len(xlags_) == Kx), "Length of 'xlags' does not coincide with total number of Y covariates"

        # Highest frequency
        Fmax = np.max(freq_)

        # Sample slicing indexes 
        tau2 = np.sum(np.maximum.reduce(np.isnan(Zt), axis=1))

        if debug:
            print("---")
            print(f" tau2  =  {tau2}")
            print("---")

        # Generate slicing indexes 
        T = len(Yt)
        now_zslices = []
        now_yslices = []
        for r in range(Fmax):
            # NOTE: handle cases where 'NaN' observations from high-freq.
            #       lags must be excluded in the first block
            o = (Fmax + r) if (r < tau2) else r
            #
            now_zslices.append(np.arange(o, T, Fmax))
            now_yslices.append(np.arange(o, T, Fmax))

        if debug:
            print("Regressors slicing indexes [now_zslices]:")
            for r in range(Fmax):
                print(f"r = {r+1}: {now_zslices[r]}")
            print("Response slicing indexes [now_yslices]:")
            print("[same as now_zslices]")

        # Define weight function
        if normalize:
            def phi_MIDAS(theta):
                phi = np.zeros(1+np.sum(ylags_)+np.sum(xlags_+1))
                p = 1+np.sum(ylags_)
                phi[0:p] = theta[0:p]
                # Almond lag parameters
                m = p
                for k in range(Kx):
                    q_k = xlags_[k]+1
                    phi_k = almon(q_k, theta[p+3*k], theta[p+3*k+1], theta[p+3*k+2])
                    phi[m:(m+q_k)] = phi_k
                    m += q_k
                return phi
        
        else:
            def phi_MIDAS(theta):
                phi = np.zeros(1+np.sum(ylags_)+np.sum(xlags_+1))
                p = 1+np.sum(ylags_)
                phi[0:p] = theta[0:p]
                # Almond lag parameters
                m = p
                for k in range(Kx):
                    q_k = xlags_[k]+1
                    phi_k = almon_nonorm(q_k, theta[p+3*k], theta[p+3*k+1], theta[p+3*k+2])
                    phi[m:(m+q_k)] = phi_k
                    m += q_k
                return phi

        # Init parameter vector
        theta0     = np.zeros(1+np.sum(ylags_)+(Kx*3)) 
        #theta0     = 0.001 * np.ones(1+np.sum(ylags_)+(Kx*3))

        # Optimize
        if method == 'nls':
            def Res_obj(theta, i):
                    return np.squeeze(Yt[now_yslices[i],:] - Zt[now_zslices[i],:] @ np.vstack(phi_MIDAS(theta)))

            res = []
            for r in range(Fmax):
                verbose = 2 if debug else 1
                res_r = optim.least_squares(lambda _theta_ : Res_obj(_theta_, r), theta0, 
                            max_nfev=10000, ftol=1e-8, xtol=1e-8, gtol=1e-8, method='lm', verbose=verbose)
                
                res.append({
                    'x':            res_r.x,
                    'fun':          np.sum(res_r.fun**2),
                    'status':       res_r.status,
                    'message':      res_r.message,
                })

                if debug:
                    print("---")
                    print(f"r = {r} / {Fmax-1} | [h = {Fmax-1-r} / {Fmax-1}]")
                    print("Joint regression matrix:")
                    with np.printoptions(precision=4, suppress=True):
                        print(np.hstack((Yt[now_yslices[r],:], Zt[now_zslices[r],:])))
                    print("NLS optimizer [theta_opt]:")
                    print(res_r.x)

                #if (res_r.status >= 0):
                #    theta0_new = np.copy(res_r.x)
                #else:
                #    theta0_new = np.copy(theta0)

        elif method == 'patternsearch':
            res = []
            for r in range(Fmax):
                def RSS_obj(theta):
                    return np.sum(np.square(Yt[now_yslices[r],:] - Zt[now_zslices[r],:] @ np.vstack(phi_MIDAS(theta))))
                
                problem = FunctionalProblem(1+np.sum(ylags_)+(Kx*3),
                            RSS_obj,
                            #constr_ieq=constr_ieq,
                            x0=theta0,
                            #x0=500,
                            xl=(-1000 * np.ones(1+np.sum(ylags_)+(Kx*3))),
                            xu=(1000 * np.ones(1+np.sum(ylags_)+(Kx*3)))
                            )

                res_r = minimize(problem, PatternSearch(), verbose=True, seed=1209803)
                
                if debug:
                    print("Best solution found: \nX = %s\nF = %s" % (res_r.X, res_r.F))
                
                res.append({
                    'x':            res_r.X,
                    'fun':          res_r.F,
                    'status':       "NA",
                    'message':      "NA",
                })

        elif method == 'nelmead':
            res = []
            for r in range(Fmax):
                def RSS_obj(theta):
                    return np.sum(np.square(Yt[now_yslices[r],:] - Zt[now_zslices[r],:] @ np.vstack(phi_MIDAS(theta))))
    
                res_r = optim.minimize(RSS_obj, theta0, method='nelder-mead', 
                                options={'maxiter': 1e4, 'disp': debug, 'adaptive': True})
                
                res.append({
                    'x':            res_r.x,
                    'fun':          res[r].fun,
                    'status':       res_r.status,
                    'message':      res_r.message,
                })
        
        else:
            raise ValueError("Incorrect optimization method selected") 

        parsfit_midas_now = []
        # NOTE: convention is that nowcasting horizon 'h==0' is
        #       for contemporaneous regression, therefore
        for h in range(Fmax):
            r = (Fmax-1) - h
            # Outputs
            theta_opt_h = res[r]['x']
            phi_opt_h   = phi_MIDAS(theta_opt_h)
            RSS_h       = res[r]['fun']
            status_h    = res[r]['status']
            message_h   = res[r]['message']

            # NOTE: COMPARISON WITH MATLAB (manual debug only)
            #theta1      = 0.03 * np.ones(1+np.sum(ylags_)+(Kx*3))
            #theta_opt_h = theta1
            #phi_opt_h   = phi_MIDAS(theta1)

            Yt_fit_h    = Zt[now_zslices[r],:] @ np.vstack(phi_opt_h)
            Residuals_h = Yt[now_yslices[r],:] - Yt_fit_h

            parsfit_midas_now.append({
                'r':                r,
                'h':                h,
                'theta_opt':        theta_opt_h,
                'phi_opt':          phi_opt_h,
                'theta0':           theta0,
                'Yt_fit':           Yt_fit_h,
                'Residuals':        Residuals_h,
                'RSS':              RSS_h,
                'optim_status':     status_h,
                'optim_message':    message_h,
            })
            
        return parsfit_midas_now, now_zslices, now_yslices

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MIDAS Fixed-Parameter Nowcasting [NOWCASTING]

    def fixedparamsNowcast(self, data, fit_length, method='nls', normalize=True, grad=False, debug=False):

        # Unpack data, options
        Y, X = data

        if debug:
            print("---")
            print("MIDAS.fixedparamsNowcast() | MIDAS model [Nowcasting] -> Debug Info")

        # Get self parameters, makes expressions easier to read
        ylags = self.ylags_
        xlags = self.xlags_
        freq  = self.freq_

        # (1) Create MIDAS regression matrices ---
        Yt_full, Zt_full, mask, Tfull, Ky, Kx = self.fitAlmon_prep_data_now(Y, X, freq, ylags, xlags, debug=debug)

        # Dynamic (i.e. autoregressive) MIDAS flag
        dynamic = not ((self.ylags_ is None) or (self.ylags_ == 0))

        # Highest frequency
        Fmax = np.max(freq)

        # (2) Subset regression matrices based on 'fit_length' 
        h1 = (xlags+1) / freq
        tau0 = np.max(np.concatenate((ylags, np.ceil(h1)))).astype(int)

        # NOTE: without lags of 'Y', 'Tcutoff' can be directly
        #       simplified to 'fit_length * Fmax'
        if ylags == 0:
            Tcutoff = fit_length * Fmax
        else:
            Tcutoff = (fit_length - tau0) * Fmax

        assert (Tfull > Tcutoff), "Sample size T is too small compared to training length"

        # Fitting data
        Yt_nowfit = Yt_full[0:Tcutoff,]
        Zt_nowfit = Zt_full[0:Tcutoff,]

        # Test data
        Yt_nowtest = Yt_full[Tcutoff:,]
        Zt_nowtest = Zt_full[Tcutoff:,]

        # (3) Optimize Almon lag (+ autoregressive) parameters ---
        if grad:
            raise ValueError("TODO")
        else:
            parsfit_midas_now, now_zslices, now_yslices = self.fitAlmon_solve_now(Yt_nowfit, Zt_nowfit, Ky, Kx, freq, ylags, xlags, method=method, normalize=normalize, debug=debug)
        
        # (4) Compute nowcasts 
        nowcast_out = []
        for h in range(Fmax):
            r = (Fmax-1) - h
            zyslice = np.arange(r, len(Zt_nowtest), Fmax)

            # Compute nowcast
            phi_h = parsfit_midas_now[h]['phi_opt']
            Yt_nowtest_h    = Zt_nowtest[zyslice,:] @ np.vstack(phi_h)
            Error_nowtest_h = Yt_nowtest[zyslice,:] - Yt_nowtest_h
            NESS_h          = np.sum(Error_nowtest_h**2)

            if debug:
                print("---")
                print(f"r = {r} | [h = {h} / {Fmax-1}]")
                print("Joint nowcast matrix:")
                with np.printoptions(precision=4, suppress=True):
                    print(np.hstack((Yt_nowtest[zyslice,:], Zt_nowtest[zyslice,:])))

            #print(parsfit_midas_now[h])

            #
            nowcast_out.append({
                'r':            r,
                'h':            h,
                'Nowcast':      Yt_nowtest_h,
                'Errors':       Error_nowtest_h,
                'NESS':         NESS_h,
                'zy_slice':     zyslice,
                'Yt_test':      Yt_nowtest[zyslice,:],
                'Zt_test':      Zt_nowtest[zyslice,:],
            })

        # End nowcasting loop

        # Output
        nowcast_out = {
            'parsfit_now':  parsfit_midas_now,
            'nowcasts':     nowcast_out,
            'z_slices':     now_zslices,
            'y_slices':     now_yslices,
            'freq':         freq,
            'ylags':        ylags,
            'xlags':        xlags,
            'Ky':           Ky,
            'Kx':           Kx,
            'mask':         mask,
            'method':       method,
            'dynamic':      dynamic,
            'normalize':    normalize,
            'type':         'Almon',
            'grad':         grad,
        }
        
        return nowcast_out

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Helpers

    def fitted_reshape(self, fit_out, steps=None, T=None):
        # Reshape a 'fit_out' output object into a list of vectors

        #Fmax  = np.max(np.array(freq))
        steps = len(fit_out) if (steps is None) else steps
        T     = steps * len(fit_out[0]['Yt_fit']) if (T is None) else T

        Yt_is  = np.full((T, 1), np.nan)
        Yt_fit = np.full((T, 1), np.nan)
        Res    = np.full((T, 1), np.nan)
        for s in range(steps):
            T_s   = len(fit_out[s]['Yt_fit'])
            idx_s = range(T-(T_s-1)*steps-1-s, T, steps)
            # Sample data
            Yt_is[idx_s] = fit_out[s]['Yt_fit'] + fit_out[s]['Residuals']
            # Sample fit
            Yt_fit[idx_s] = fit_out[s]['Yt_fit']
            # Sample residuals
            Res[idx_s] = fit_out[s]['Residuals']

        # Output
        reshape_out = {
            'T':                T,
            'Yt_is':            Yt_is,
            'Yt_fit':           Yt_fit,
            'Residuals':        Res,
        }

        return reshape_out

    def forecast_reshape(self, for_fit_out, cut_steps=True):
        # Reshape the forecast output from 'fixedparamsForecast' into a vector

        Fmax  = np.max(for_fit_out['freq'])
        steps = for_fit_out['steps']
        L     = steps if cut_steps else Fmax
        Tos   =  L * len(for_fit_out['forecasts'][0]['Forecast'])

        # Pre-allocate
        Yt_os = np.full((Tos, 1), np.nan)
        Yt_forecast = np.full((Tos, 1), np.nan)
        Yt_os_error = np.full((Tos, 1), np.nan)
        for j in range(steps):
            slice_j = range((L-1) - j, Tos, L)
            # Out-of-sample values
            Yt_os[slice_j] = for_fit_out['forecasts'][0]['Yt_test']
            # Out-of-sample forecast
            Yt_forecast[slice_j] = for_fit_out['forecasts'][j]['Forecast']
            # Out-of-sample error
            Yt_os_error[slice_j] = for_fit_out['forecasts'][j]['Errors']

        # Output
        reshape_out = {
            'Tos':              Tos,
            'Yt_os':            Yt_os,
            'Yt_forecast':      Yt_forecast,
            'Yt_os_error':      Yt_os_error,
        }

        return reshape_out

    def nowcast_reshape(self, now_fit_out, cut_period=None):
        # Reshape the nowcast output from 'fixedparamsNowcast' into vector

        if not (cut_period is None):
            assert type(cut_period) is int

        Fmax  = np.max(now_fit_out['freq'])
        # period = now_fit_out['period']
        L     = cut_period if (not cut_period is None) else Fmax
        Tos   = L * len(now_fit_out['nowcasts'][0]['Nowcast'])
        #Tns   = Fmax * len(now_fit_out['nowcasts'][0]['Nowcast'])

        # Pre-allocate
        Yt_os = np.full((Tos, 1), np.nan)
        Yt_nowcast  = np.full((Tos, 1), np.nan)
        Yt_os_error = np.full((Tos, 1), np.nan)
        for j in range(L):
            slice_j = range((L-1) - j, Tos, L)
            # Out-of-sample values
            Yt_os[slice_j] = now_fit_out['nowcasts'][0]['Yt_test']
            # Out-of-sample forecast
            Yt_nowcast[slice_j] = now_fit_out['nowcasts'][j]['Nowcast']
            # Out-of-sample error
            Yt_os_error[slice_j] = now_fit_out['nowcasts'][j]['Errors']

        # Output
        reshape_out = {
            'Tos':              Tos,
            'Yt_os':            Yt_os,
            'Yt_nowcast':       Yt_nowcast,
            'Yt_os_error':      Yt_os_error,
        }

        return reshape_out

    def plotFitted(self, fit_out, col='C1', figsize=(10, 4)):
        midas_fits = None
        midas_data = None
        if 'parsfit_for' in fit_out:
            #midas_fits = self.getFitted_fixparfor(fit_out)[j][:,1]
            #midas_data = fit_out['Y']
            #midas_data = midas_data[midas_data != np.inf]

            reshaped_fit = self.fitted_reshape(fit_out['parsfit_for'], fit_out['steps'])
            midas_fits = reshaped_fit['Yt_fit']
            midas_data = reshaped_fit['Yt_is']

        elif 'parsfit_now' in fit_out:
            reshaped_fit = self.fitted_reshape(fit_out['parsfit_now'], np.max(fit_out['freq']))
            midas_fits = reshaped_fit['Yt_fit']
            midas_data = reshaped_fit['Yt_is']

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(midas_data, label="Data")
        plt.plot(midas_fits, label="Fitted", color=col)
        plt.grid()
        ax.legend()
        ax.set_title("MIDAS - Fitted Values")
        #return fig

    def plotResiduals(self, fit_out, col='C4', figsize=(10, 4)):
        midas_resd = None
        #midas_data = None
        if 'parsfit_for' in fit_out:
            #midas_resd = self.getResiduals_fixparfor(fit_out)[j][:,1]
            
            reshaped_fit = self.fitted_reshape(fit_out['parsfit_for'], fit_out['steps'])
            midas_resd = reshaped_fit['Residuals']
            midas_fits = reshaped_fit['Yt_fit']
            midas_data = reshaped_fit['Yt_is']

        elif 'parsfit_now' in fit_out:
            reshaped_fit = self.fitted_reshape(fit_out['parsfit_now'], np.max(fit_out['freq']))
            midas_resd = reshaped_fit['Residuals']
            midas_data = reshaped_fit['Yt_is']

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(midas_data, label="Data", alpha=0.3)
        plt.plot(midas_resd, label="Residuals", color=col)
        plt.grid()
        ax.legend()
        ax.set_title("MIDAS - Fit Residuals")
        #return fig

    def plotPhi(self, fit_out, i=0, figsize=(10, 4)):
        midas_phi_opt = None
        ylags = fit_out['ylags']
        xlags = fit_out['xlags']
        Kx = fit_out['Kx']
        #
        if 'parsfit_for' in fit_out:
            midas_phi_opt = fit_out['parsfit_for'][i]['phi_opt']
        elif 'parsfit_now' in fit_out:
            midas_phi_opt = fit_out['parsfit_now'][i]['phi_opt']

        phi = midas_phi_opt
        fig, ax = plt.subplots(figsize=figsize)
        plt.axhline(c="k", ls="--", lw=.7)
        plt.plot(0, phi[0], marker="o")
        p = np.sum(ylags)+1
        plt.plot(range(1, p), phi[1:p], marker="o")
        m = p
        for k in range(Kx):
            if (type(xlags) is int):
                idx_k = xlags
            elif (len(xlags) == 1):
                idx_k = xlags[0]
            else:
                idx_k = xlags[k]
            plt.plot(range(m, (m+(idx_k+1))), phi[m:(m+(idx_k+1))], marker="o")
            m += (idx_k+1)
        plt.grid()
        ax.set_title("MIDAS - Fitted Parameters")
        #return fig

    def plotFixedParamsForecasts(self, for_fit_out, col='C2', cut_steps=True, figsize=(10, 4)):
        midas_for_fit_out = self.forecast_reshape(for_fit_out, cut_steps=cut_steps)
        midas_data = midas_for_fit_out['Yt_os']
        midas_forecasts = midas_for_fit_out['Yt_forecast']

        steps = for_fit_out['steps']

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(midas_data, label="Data", alpha=0.3)
        plt.plot(midas_forecasts, label="Forecast", color=col)
        plt.grid()
        ax.legend()
        ax.set_title(f"MIDAS - Fixed Parameter Forecast ({steps} steps)")
        #return 0

    def plotFixedParamsNowcasts(self, now_fit_out, col='C2', cut_period=None, figsize=(10, 4)):
        midas_now_fit_out = self.nowcast_reshape(now_fit_out, cut_period=cut_period)
        midas_data = midas_now_fit_out['Yt_os']
        midas_nowcasts = midas_now_fit_out['Yt_nowcast']

        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(midas_data, label="Data", alpha=0.3)
        plt.plot(midas_nowcasts, label="Nowcast", color=col)
        plt.grid()
        ax.legend()
        ax.set_title(f"MIDAS - Fixed Parameter Nowcast")
        #return 0

        