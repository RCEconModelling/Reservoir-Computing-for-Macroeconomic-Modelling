###########
# Quarterly-Monthly-Daily dynamic linear factor model using Kalman filtering
###########

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
tfd = tfp.distributions
import pandas as pd
import pdb

class QMWDDFM:
  def __init__(self, q_process_dim, m_process_dim, d_process_dim, map_timestamp_to_model, order=1, lags=1,
               latent_process_dim = 10, w_process_dim = 0, dtype = tf.float32, non_mask_ar = None,
               q_lags = 1, m_lags = 1, w_lags = 1, d_lags = 1, q_freqs = None, m_freqs = None, w_freqs = 1, d_freqs = 1,
               model = 'stock', lambda_init=1, contractive = False):
    self._latent_process_dim = latent_process_dim
    self._order = order
    if model == 'stock':
      q_lags = 1
      m_lags = 1
      w_lags = 1
      d_lags = 1
    if model == 'freq' and q_freqs is None:
      q_freqs = 5
    if model == 'freq' and m_freqs is None:
      m_freqs = 3
    self._lags = max(lags, order, q_lags, w_lags, m_lags, d_lags)
    if non_mask_ar is not None:
      self._lags = max(self._lags, np.max(non_mask_ar))
    self._map_timestamp_to_model = map_timestamp_to_model
    self._params = []
    self._q_process_dim = q_process_dim
    self._m_process_dim = m_process_dim
    self._w_process_dim = w_process_dim
    self._d_process_dim = d_process_dim
    self._q_lags = q_lags
    self._m_lags = m_lags
    self._w_lags = w_lags
    self._d_lags = d_lags
    self._non_mask_ar = non_mask_ar
    self._q_freqs = q_freqs
    self._m_freqs = m_freqs
    self._w_freqs = w_freqs
    self._d_freqs = d_freqs
    self._C = 25. #constant so its eponential is large
    self._dtype = dtype
    self._model = model
    self._contractive = contractive
    self._lambda_init = lambda_init #initial diagonal of latent transition matrix
    self.init_model()


  def init_model(self):
    self.build_observation_matrix()
    self._observation_log_std_q = tf.Variable(2 * tf.ones([self._q_process_dim], dtype = self._dtype), trainable = True)
    self._observation_log_std_m = tf.Variable(2 * tf.ones([self._m_process_dim], dtype = self._dtype), trainable = True)
    self._observation_log_std_d = tf.Variable(2 * tf.ones([self._d_process_dim], dtype = self._dtype), trainable = True)
    self._params.append(self._observation_log_std_q)
    self._params.append(self._observation_log_std_m)
    self._params.append(self._observation_log_std_d)

    if self._w_process_dim > 0:
      self._observation_log_std_w = tf.Variable(tf.ones([self._w_process_dim], dtype = self._dtype), trainable = True)
      self._params.append(self._observation_log_std_w)

    # params for the transition model
    if self._order == 1 and self._lags == 1:
      self._transition_matrices = [tf.Variable(
        self._lambda_init * tf.random.normal([self._latent_process_dim, self._latent_process_dim], dtype = self._dtype), trainable = True)]
      self._transition_log_std = tf.Variable(2 * tf.ones([self._latent_process_dim], dtype = self._dtype), trainable = True)
      self._params.append(self._transition_matrices[0])
      self._params.append(self._transition_log_std)
    else:
      if self._non_mask_ar is None:
        self._transition_matrices = [tf.Variable(
          self._lambda_init * tf.random.normal([self._latent_process_dim, self._latent_process_dim],
                                               dtype = self._dtype),
          trainable = True) for _ in range(self._order)]
      else:
        self._transition_matrices = [tf.Variable(
          self._lambda_init * tf.random.normal([self._latent_process_dim, self._latent_process_dim],
                                               dtype = self._dtype),
          trainable = True) for _ in range(len(self._non_mask_ar))]
      self._params.extend(self._transition_matrices)
      self._transition_log_std = tf.Variable(2 * tf.ones([self._latent_process_dim], dtype = self._dtype), trainable = True)
      self._params.append(self._transition_log_std)

    #transition_matrix = self.make_transition_matrix_fn(0)
    #init_transition_eigenvalue, _ = tf.linalg.eigh(transition_matrix)
    #self._transition_eigenvalue_normalisation = self._lambda_init * tf.constant(
    #    tf.reduce_max(abs(init_transition_eigenvalue)))
    #print(self._transition_eigenvalue_normalisation)

    # transition model
    def transition_matrix_fn(t):
      transition_matrix = self.make_transition_matrix_fn(t)
      if not self._contractive:
        return tf.linalg.LinearOperatorFullMatrix(matrix = transition_matrix)
      else:
        assert([1] == self._non_mask_ar)
        eps = tf.cast(1. - 1e-4, self._dtype)
        A0 = transition_matrix[:self._latent_process_dim, :self._latent_process_dim]
        eig_A0 = tf.reduce_max(abs(tf.linalg.eigh(A0)[0]))
        A0_adjusted = A0 * eps / tf.maximum(eps,eig_A0)
        transition_matrices_adjusted = [A0_adjusted] + [
          tf.zeros([self._latent_process_dim, self._latent_process_dim], dtype = self._dtype)
          for _ in range(self._lags - self._order)]
        if self._lags == 1:
          transition_matrix_adjusted = transition_matrices_adjusted[0]
        else:
          transition_matrix_adjusted = make_ar_transition_matrix(transition_matrices_adjusted, dtype = self._dtype)
        return tf.linalg.LinearOperatorFullMatrix(matrix = transition_matrix_adjusted)

    def transition_noise_fn(t):
      if self._lags > 1:
        transition_log_std = tf.concat(
          [self._transition_log_std] + [
            -tf.constant(np.inf, dtype = self._dtype) * tf.ones_like(self._transition_log_std, dtype = self._dtype)
          ] * (self._lags - 1), 0)
      else:
        transition_log_std = self._transition_log_std
      return tfd.MultivariateNormalDiag(scale_diag = tf.math.exp(transition_log_std))



    def observation_matrix_fn(t):
      if self._model == 'stock':
        if self._w_process_dim == 0:
          observation_matrix_q = tf.concat([tf.concat([q, m, d], 0) for (q, m, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_d)], 1)
          observation_matrix_m = tf.concat([tf.concat([tf.zeros_like(q), m, d], 0) for (q, m, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_d)], 1)
          observation_matrix_d = tf.concat([tf.concat([tf.zeros_like(q), tf.zeros_like(m), d], 0) for (q, m, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_d)], 1)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 0,
                                       lambda: observation_matrix_d,
                                       lambda: observation_matrix_m)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 2,
                                       lambda: observation_matrix_q,
                                       lambda: observation_matrix)
        else:
          observation_matrix_q_w = tf.concat([tf.concat([q, m, w, d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_m_w = tf.concat([tf.concat([tf.zeros_like(q), m, w, d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_d_w = tf.concat(
            [tf.concat([tf.zeros_like(q), tf.zeros_like(m), w, d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)

          observation_matrix_q = tf.concat([tf.concat([q, m, tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_m = tf.concat(
            [tf.concat([tf.zeros_like(q), m, tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)
          observation_matrix_d = tf.concat(
            [tf.concat([tf.zeros_like(q), tf.zeros_like(m), tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)

          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 0,
                                       lambda: observation_matrix_d,
                                       lambda: observation_matrix_m)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 2,
                                       lambda: observation_matrix_q,
                                       lambda: observation_matrix)

          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 3,
                                       lambda: observation_matrix_d_w,
                                       lambda: observation_matrix)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 4,
                                       lambda: observation_matrix_m_w,
                                       lambda: observation_matrix)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 5,
                                       lambda: observation_matrix_q_w,
                                       lambda: observation_matrix)

        return tf.linalg.LinearOperatorFullMatrix(matrix = observation_matrix)

      elif self._model in ['almon', 'freq']:

        if self._model == 'almon':

          def almon(K, theta1, theta2, beta):
            f1 = tf.math.exp(theta1[:, :, None] * tf.range(K, dtype = self._dtype))
            f2 = tf.math.exp(theta2[:, :, None] * tf.range(K, dtype = self._dtype) * tf.range(K, dtype = self._dtype))
            unnorm_weights = f1 * f2
            return beta[:, :, None] * unnorm_weights / tf.reduce_sum(unnorm_weights, -1)[:, :, None]


          observation_matrix_m_ = almon(self._m_lags, self._theta_m[0], self._theta_m[1], self._beta_m)
          observation_matrix_q_ = almon(self._q_lags, self._theta_q[0], self._theta_q[1], self._beta_q)
          observation_matrix_d_ = almon(self._d_lags, self._theta_d[0], self._theta_d[1], self._beta_d)

        elif self._model == 'freq':

          def vand_matrix(L, freqs, diags, offset, readout):
            freqs = tf.nn.sigmoid(freqs)
            #tf.print('freqs', freqs)
            A = diags * diags * tf.cos(
              2. * tf.cast(np.pi, self._dtype) * freqs[None] * (tf.range(L, dtype = self._dtype))[:,None] + offset)
            A = tf.reduce_sum(A, -1)
            temperature = tf.math.sqrt(tf.ones([1], A.dtype) *freqs.shape[0])
            B = tf.math.exp(A/temperature) / tf.reduce_sum(tf.math.exp(A/temperature), 0)
            #tf.print('B', B)
            return tf.einsum('ij,l->ijl', readout, B)

          observation_matrix_q_ = vand_matrix(self._q_lags, self._vand_freq_q, self._vand_diag_q, self._vand_offset_q,
                                              self._vand_readout_q)
          observation_matrix_m_ = vand_matrix(self._m_lags, self._vand_freq_m, self._vand_diag_m, self._vand_offset_m,
                                              self._vand_readout_m)
          observation_matrix_d_ = vand_matrix(self._d_lags, self._vand_freq_d, self._vand_diag_d, self._vand_offset_d,
                                              self._vand_readout_d)

          #tf.print('self._vand_freq_q', self._vand_freq_q)
          #tf.print('self._vand_diag_q', self._vand_diag_q)
          #tf.print('self._vand_offset_q', self._vand_offset_q)
          #tf.print('self._vand_readout_q', self._vand_readout_q)
          #tf.print('self._vand_freq_m', self._vand_freq_m)
          #tf.print('self._vand_diag_m', self._vand_diag_m)


        observation_matrix_q_ = tf.concat([observation_matrix_q_, tf.zeros(
          [self._q_process_dim, self._latent_process_dim, self._lags - self._q_lags], dtype = self._dtype)], -1)
        observation_matrix_m_ = tf.concat([observation_matrix_m_, tf.zeros(
          [self._m_process_dim, self._latent_process_dim, self._lags - self._m_lags], dtype = self._dtype)], -1)
        observation_matrix_d_ = tf.concat([observation_matrix_d_, tf.zeros(
          [self._d_process_dim, self._latent_process_dim, self._lags - self._d_lags], dtype = self._dtype)], -1)

        observation_matrix_q_ = tf.reshape(observation_matrix_q_, [self._q_process_dim, -1])
        observation_matrix_m_ = tf.reshape(observation_matrix_m_, [self._m_process_dim, -1])
        observation_matrix_d_ = tf.reshape(observation_matrix_d_, [self._d_process_dim, -1])



        if self._w_process_dim == 0:
          observation_matrix_q = tf.concat([
            observation_matrix_q_, observation_matrix_m_, observation_matrix_d_], 0)
          observation_matrix_m = tf.concat([
            tf.zeros_like(observation_matrix_q_), observation_matrix_m_, observation_matrix_d_], 0)
          observation_matrix_d = tf.concat([
            tf.zeros_like(observation_matrix_q_), tf.zeros_like(observation_matrix_m_), observation_matrix_d_], 0)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 0,
                                       lambda: observation_matrix_d,
                                       lambda: observation_matrix_m)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 2,
                                       lambda: observation_matrix_q,
                                       lambda: observation_matrix)
        else:
          observation_matrix_q_w = tf.concat([tf.concat([q, m, w, d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_m_w = tf.concat([tf.concat([tf.zeros_like(q), m, w, d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_d_w = tf.concat(
            [tf.concat([tf.zeros_like(q), tf.zeros_like(m), w, d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)

          observation_matrix_q = tf.concat([tf.concat([q, m, tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
            self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
            self._observation_matrices_d)], 1)
          observation_matrix_m = tf.concat(
            [tf.concat([tf.zeros_like(q), m, tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)
          observation_matrix_d = tf.concat(
            [tf.concat([tf.zeros_like(q), tf.zeros_like(m), tf.zeros_like(w), d], 0) for (q, m, w, d) in zip(
              self._observation_matrices_q, self._observation_matrices_m, self._observation_matrices_w,
              self._observation_matrices_d)], 1)

          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 0,
                                       lambda: observation_matrix_d,
                                       lambda: observation_matrix_m)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 2,
                                       lambda: observation_matrix_q,
                                       lambda: observation_matrix)

          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 3,
                                       lambda: observation_matrix_d_w,
                                       lambda: observation_matrix)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 4,
                                       lambda: observation_matrix_m_w,
                                       lambda: observation_matrix)
          observation_matrix = tf.cond(tf.expand_dims(self.calender[t] * tf.ones_like(t), 0) == 5,
                                       lambda: observation_matrix_q_w,
                                       lambda: observation_matrix)

        return tf.linalg.LinearOperatorFullMatrix(matrix = observation_matrix)



    self._observation_matrix_fn = observation_matrix_fn



    def observation_noise_fn(t):
      if self._w_process_dim == 0:

        observation_scale_q = tf.concat([
          tf.math.exp(self._observation_log_std_q),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_m = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_d = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_m)),
          tf.math.exp(self._observation_log_std_d)], 0)

        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 0,
                             lambda: observation_scale_d,
                             lambda: observation_scale_m)
        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 2,
                             lambda: observation_scale_q,
                             lambda: scale_diag)

      else:
        observation_scale_q = tf.concat([
          tf.math.exp(self._observation_log_std_q),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_w)),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_m = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_w)),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_d = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_w)),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_m)),
          tf.math.exp(self._observation_log_std_d)], 0)

        observation_scale_q_w = tf.concat([
          tf.math.exp(self._observation_log_std_q),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._observation_log_std_w),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_m_w = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._observation_log_std_m),
          tf.math.exp(self._observation_log_std_w),
          tf.math.exp(self._observation_log_std_d)], 0)
        observation_scale_d_w = tf.concat([
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_q)),
          tf.math.exp(self._observation_log_std_w),
          tf.math.exp(self._C * tf.ones_like(self._observation_log_std_m)),
          tf.math.exp(self._observation_log_std_d)], 0)

        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 0,
                           lambda: observation_scale_d,
                           lambda: observation_scale_m)
        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 2,
                                 lambda: observation_scale_q,
                                 lambda: scale_diag)

        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 3,
                           lambda: observation_scale_d_w,
                           lambda: scale_diag)
        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 4,
                           lambda: observation_scale_m_w,
                           lambda: scale_diag)
        scale_diag = tf.cond(self.calender[t] * tf.ones_like(t) == 5,
                           lambda: observation_scale_q_w,
                           lambda: scale_diag)

      noise_dist = tfd.MultivariateNormalDiag(scale_diag = scale_diag)
      return noise_dist

    self._observation_noise_fn = observation_noise_fn
    self._transition_matrix_fn = transition_matrix_fn
    self._transition_noise_fn = transition_noise_fn




  def build_observation_matrix(self):
    if self._model == 'stock':
      observation_matrices_q_param = tf.Variable(tf.random.normal([self._q_process_dim, self._latent_process_dim],
                                                                 dtype = self._dtype), trainable = True)
      self._params.append(observation_matrices_q_param)
      self._observation_matrices_q = [observation_matrices_q_param] + [
        tf.zeros([self._q_process_dim, self._latent_process_dim], dtype = self._dtype) for _ in range(self._lags - 1)]
      observation_matrices_m_param = tf.Variable(tf.random.normal([self._m_process_dim, self._latent_process_dim],
                                                                 dtype = self._dtype), trainable = True)
      self._params.append(observation_matrices_m_param)
      self._observation_matrices_m = [observation_matrices_m_param] + [
        tf.zeros([self._m_process_dim, self._latent_process_dim], dtype = self._dtype) for _ in range(self._lags - 1)]
      observation_matrices_d_param = tf.Variable(tf.random.normal([self._d_process_dim, self._latent_process_dim],
                                                                 dtype = self._dtype), trainable = True)
      self._params.append(observation_matrices_d_param)
      self._observation_matrices_d = [observation_matrices_d_param] + [
        tf.zeros([self._d_process_dim, self._latent_process_dim], dtype = self._dtype) for _ in range(self._lags - 1)]

      if self._w_process_dim > 0:
        observation_matrices_w_param = tf.Variable(tf.random.normal([self._w_process_dim, self._latent_process_dim],
                                                                   dtype = self._dtype), trainable = True)
        self._params.append(observation_matrices_w_param)
        self._observation_matrices_w = [observation_matrices_w_param] + [
        tf.zeros([self._w_process_dim, self._latent_process_dim],  dtype = self._dtype) for _ in range(self._lags - 1)]
        self._params.extend(self._observation_matrices_w)


    elif self._model == 'almon':
      # Almon lag functions
      self._theta_q = tf.Variable(tf.random.normal([2, self._q_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._beta_q = tf.Variable(tf.random.normal([self._q_process_dim, self._latent_process_dim], dtype = self._dtype),
                           trainable = True)
      self._theta_m = tf.Variable(tf.random.normal([2, self._m_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._beta_m = tf.Variable(tf.random.normal([self._m_process_dim, self._latent_process_dim], dtype = self._dtype),
                           trainable = True)
      self._theta_d = tf.Variable(tf.random.normal([2, self._d_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._beta_d = tf.Variable(tf.random.normal([self._d_process_dim, self._latent_process_dim], dtype = self._dtype),
                           trainable = True)

      self._params.append(self._theta_q)
      self._params.append(self._theta_m)
      self._params.append(self._theta_d)
      self._params.append(self._beta_q)
      self._params.append(self._beta_m)
      self._params.append(self._beta_d)


    elif self._model == 'freq':
      # freq in latent space
      self._vand_freq_q = tf.Variable(tf.random.normal([self._q_freqs], dtype = self._dtype) - 2.,
                            trainable = True)
      self._vand_diag_q = tf.Variable(tf.random.normal([self._q_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_offset_q = tf.Variable(tf.random.normal([self._q_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_readout_q = tf.Variable(tf.random.normal([self._q_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._vand_freq_m = tf.Variable(tf.random.normal([self._m_freqs], dtype = self._dtype) - 2.,
                            trainable = True)
      self._vand_diag_m = tf.Variable(tf.random.normal([self._m_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_offset_m = tf.Variable(tf.random.normal([self._m_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_readout_m = tf.Variable(tf.random.normal([self._m_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._vand_freq_d = tf.Variable(tf.random.normal([self._d_freqs], dtype = self._dtype) - 2.,
                            trainable = True)
      self._vand_diag_d = tf.Variable(tf.random.normal([self._d_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_offset_d = tf.Variable(tf.random.normal([self._d_freqs], dtype = self._dtype),
                            trainable = True)
      self._vand_readout_d = tf.Variable(tf.random.normal([self._d_process_dim, self._latent_process_dim], dtype = self._dtype),
                            trainable = True)
      self._params.append(self._vand_freq_q)
      self._params.append(self._vand_freq_m)
      self._params.append(self._vand_freq_d)
      self._params.append(self._vand_diag_q)
      self._params.append(self._vand_diag_m)
      self._params.append(self._vand_diag_d)
      self._params.append(self._vand_offset_q)
      self._params.append(self._vand_offset_m)
      self._params.append(self._vand_offset_d)
      self._params.append(self._vand_readout_q)
      self._params.append(self._vand_readout_m)
      self._params.append(self._vand_readout_d)


  def make_calender(self, dates):
    calender = tf.convert_to_tensor([self._map_timestamp_to_model(t) for t in dates])
    return calender

  def make_model(self, T):

    initial_factor_chol = tf.eye(self._latent_process_dim * self._lags, dtype = self._dtype)
    i = tf.constant(1)
    c = lambda i, initial_factor_chol: tf.less(i, self._lags)
    b = lambda i, initial_factor_chol: (i+1, self._transition_matrix_fn(0).matmul(initial_factor_chol))
    _, initial_factor_chol = tf.while_loop(c, b, [i, initial_factor_chol])
    return tfd.LinearGaussianStateSpaceModel(
      num_timesteps = T,
      transition_matrix = self._transition_matrix_fn,
      transition_noise = self._transition_noise_fn,
      observation_matrix = self._observation_matrix_fn,
      observation_noise = self._observation_noise_fn,
      initial_state_prior = tfd.MultivariateNormalTriL(scale_tril = initial_factor_chol),
      experimental_parallelize = False)

  def fit(self, y_train, train_dates, steps = 2000, learning_rate = .01):
    y_train = tf.cast(y_train, self._dtype)
    self.calender = self.make_calender(train_dates)
    optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
    #self.init_model()
    @tf.function(autograph = False)
    def m_step():
      with tf.GradientTape(persistent = False) as tape:
        tape.watch(self._params)
        model = self.make_model(y_train.shape[0])
        log_likelihood, _, _, _, _, _, _ = model.forward_filter(y_train)
        loss = -tf.reduce_sum(log_likelihood)
      grads = (tape.gradient(loss, self._params))
      optimizer.apply_gradients(zip(grads, self._params))
      tf.print(loss)

    for _ in range(steps):
      m_step()

  def predict(self, y_test, predict_dates, horizon):
    y_test = tf.cast(y_test, self._dtype)
    #update calender
    self.calender = self.make_calender(predict_dates)
    #remake model with updated calender
    model = self.make_model(y_test.shape[0])
    _, filtered_means, _, _, _, _, _ = model.forward_filter(y_test)
    predictions = []
    for t in range(y_test.shape[0] - horizon):
      predictive_means = filtered_means[t]
      for h in range(1,horizon+1):
        predictive_means = tf.linalg.matvec(self._transition_matrix_fn(t+h), predictive_means)
      predictions.append(tf.linalg.matvec(self._observation_matrix_fn(t+h), predictive_means).numpy())

    index = pd.MultiIndex.from_tuples(zip(predict_dates[:-horizon], predict_dates[h:t+h+1]),
                                      names = ["Pred Date", "Target Date"])
    return pd.DataFrame(predictions, index = index)
    #return pd.DataFrame(predictions, index=predict_dates[h:t+h+1])


  def predict_graph(self, y_test, predict_dates, horizon):
    y_test = tf.cast(y_test, self._dtype)
    #add zero y_test values, which are not used in comuting the filtered states
    y_test = tf.concat([y_test, tf.zeros([horizon, y_test.shape[1]], dtype=self._dtype)], axis=0)
    #update calender
    self.calender = self.make_calender(predict_dates)
    @tf.function(experimental_relax_shapes=True)
    def pred_fn():
      # remake model with updated calender
      model = self.make_model(y_test.shape[0])
      _, filtered_means, _, _, _, _, _ = model.forward_filter(y_test)
      predictions = tf.TensorArray(size=y_test.shape[0]-horizon, dtype=self._dtype)

      def update_mean(p, t):
        h = 1
        h, p = tf.while_loop(
          cond =  lambda h,p: h < horizon + 1,
          body = lambda h,p: (h+1, tf.linalg.matvec(self._transition_matrix_fn(t+h), p)),
          loop_vars = (h,p)
        )
        return h, p

      def body(t, predictions):
        predictive_means = filtered_means[t]
        _, updated_means = update_mean(predictive_means, t)
        y_mean = tf.linalg.matvec(self._observation_matrix_fn(t+horizon), updated_means)
        predictions = predictions.write(t, y_mean)
        return (t + 1, predictions)

      t = 0
      t, predictions = tf.while_loop(
        cond = lambda s,p: s < (y_test.shape[0] - horizon),
        body = body,
        loop_vars = (t, predictions))

      return predictions.stack()

    predictions = pred_fn()
    index = pd.MultiIndex.from_tuples(zip(predict_dates[:-horizon],
                                          predict_dates[horizon:]),
                                      names = ["Pred Date", "Target Date"])

    return pd.DataFrame(predictions.numpy(), index = index)
    #return pd.DataFrame(predictions, index=predict_dates[h:t+h+1])



  def predict_multiple(self, t0, y_test, predict_dates, horizon):
    y_test = tf.cast(y_test, self._dtype)
    #add zero y_test values, which are not used in comuting the filtered states
    y_test = tf.concat([y_test, tf.zeros([horizon, y_test.shape[1]], dtype=self._dtype)], axis=0)
    #update calender
    self.calender = self.make_calender(predict_dates)
    @tf.function(experimental_relax_shapes=True)
    def pred_fn():
      # remake model with updated calender
      model = self.make_model(y_test.shape[0])
      _, filtered_means, _, _, _, _, _ = model.forward_filter(y_test)
      y_dim = self._q_process_dim + self._m_process_dim + self._w_process_dim + self._d_process_dim
      factor_predictions = tf.TensorArray(size=y_test.shape[0]-horizon-t0, dtype=self._dtype,
                                          element_shape = [horizon+1, filtered_means.shape[-1]])
      observation_predictions = tf.TensorArray(size=y_test.shape[0]-horizon-t0, dtype=self._dtype,
                                               element_shape = [horizon+1, y_dim])
      def make_predictions(t,f):

        pred_state = tf.TensorArray(size=horizon+1, dtype=self._dtype)
        pred_obs = tf.TensorArray(size=horizon+1, dtype=self._dtype)
        pred_state = pred_state.write(0, f)
        y_pred = tf.linalg.matvec(self._observation_matrix_fn(t), f)
        pred_obs = pred_obs.write(0, y_pred)
        h = tf.zeros([], dtype = tf.int32)
        def update_body(h, ps, po, f0):
          #f0 = ps.gather(h)
          h = h + 1
          ps_new = tf.linalg.matvec(self._transition_matrix_fn(t+h), f0)
          po_new = tf.linalg.matvec(self._observation_matrix_fn(t+h), ps_new)
          ps = ps.write(h, ps_new)
          po = po.write(h, po_new)
          return h, ps, po, ps_new

        h, pred_state, pred_obs, f0 = tf.while_loop(
          cond =  lambda h,ps,po, f0: h < horizon ,
          body = update_body,
          loop_vars = (h, pred_state, pred_obs, f)
        )
        return pred_state.stack(), pred_obs.stack()

      def body(t, factor_pred, observation_pred):
        predicted_states, predicted_obs = make_predictions(t, filtered_means[t])
        observation_pred = observation_pred.write(t-t0, predicted_obs)
        factor_pred = factor_pred.write(t-t0, predicted_states)
        return (t + 1, factor_pred, observation_pred)

      t = t0
      t, factor_predictions, observation_predictions = tf.while_loop(
        cond = lambda s,fp,op: s < (y_test.shape[0] - horizon),
        body = body,
        loop_vars = (t, factor_predictions, observation_predictions))
      return factor_predictions.stack(), observation_predictions.stack()


    factor_predictions, observation_predictions = pred_fn()
    indices = [pd.MultiIndex.from_tuples(zip(predict_dates[t0:-horizon],predict_dates[t0+l:t0+l+factor_predictions.shape[0]]),
                                         names = ["Pred Date", "Target Date"]) for l in range(horizon+1)]
    factor_prediction_list = [pd.DataFrame(factor_predictions[:,l,:].numpy(), index = indices[l])
                              for l in range(horizon+1)]
    observation_predictions_list = [pd.DataFrame(observation_predictions[:,l,:].numpy(), index = indices[l])
                                   for l in range(horizon+1)]
    return factor_prediction_list, observation_predictions_list


  # def predicts(self, y_test, predict_dates, horizon):
  #   y_test = tf.cast(y_test, self._dtype)
  #   #update calender
  #   self.calender = self.make_calender(predict_dates)
  #   #remake model with updated calender
  #   model = self.make_model(y_test.shape[0])
  #   _, filtered_means, _, _, _, _, _ = model.forward_filter(y_test)
  #   predictions = {h:[] for h in range(1, horizon+1)}
  #   for t in range(y_test.shape[0] - horizon):
  #     predictive_means = filtered_means[t]
  #     for h in range(1,horizon+1):
  #       predictive_means = tf.linalg.matvec(self._transition_matrix_fn(t+h), predictive_means)
  #       predictions[h].append(tf.linalg.matvec(self._observation_matrix_fn(t + h), predictive_means).numpy())
  #     #predictions.append(tf.linalg.matvec(self._observation_matrix_fn(t+h), predictive_means).numpy())
  #   #return [pd.DataFrame(predictions, index=predict_dates[h:t+h+1]) for h in range(1, horizon+1)]
  #   return [pd.DataFrame(predictions[h], index=predict_dates[h:t+h+1]) for h in range(1, horizon+1)]


  def make_transition_matrix_fn(self, t):
    if self._lags > 1 and self._non_mask_ar is None:
      transition_matrices = self._transition_matrices + [
        tf.zeros([self._latent_process_dim, self._latent_process_dim], dtype = self._dtype)
      for _ in range(self._lags - self._order)]
      transition_matrix = make_ar_transition_matrix(transition_matrices, dtype = self._dtype)
    elif self._lags == 1:
      transition_matrix = self._transition_matrices[0]
    elif self._non_mask_ar is not None:
      transition_matrices = []
      j = 0
      for i in range(self._lags):
        if i+1 in self._non_mask_ar:
          transition_matrices.append(self._transition_matrices[j])
          j += 1
        else:
          transition_matrices.append(tf.zeros(
        [self._latent_process_dim, self._latent_process_dim], dtype = self._dtype))
      #transition_matrices = [self._transition_matrices[i] if i in self._non_mask_ar else tf.zeros(
      #  [self._latent_process_dim, self._latent_process_dim], dtype = self._dtype) for i in range(self._lags)]
      transition_matrix = make_ar_transition_matrix(transition_matrices, dtype = self._dtype)
    return transition_matrix


def make_ar_transition_matrix(coefficients, dtype):
  order = len(coefficients)
  dims = coefficients[0].shape[0]
  top_row = tf.concat(coefficients,-1)

  diag = tf.linalg.band_part(tf.eye((order-1)*dims, dtype = dtype),dims, 0)
  remaining_rows = tf.concat([diag,tf.zeros([dims*(order-1), dims], dtype = dtype)], axis=-1)

  ar_matrix = tf.concat([top_row, remaining_rows], axis=-2)
  return ar_matrix