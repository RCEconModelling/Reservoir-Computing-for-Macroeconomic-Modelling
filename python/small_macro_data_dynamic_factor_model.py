import os
from absl import flags
import sys
import matplotlib.pyplot as plt
import numpy as np
from qmwd_dynamic_factor_model import QMWDDFM
import tensorflow as tf
import datetime
import pandas as pd
import pdb
from pathlib import Path


flags.DEFINE_integer("horizon",
                     default=8,
                     help="forecasting horizon in hf time scale,"
                          "(nowcasting horizon until low-frequency realisation)")
flags.DEFINE_integer("latent_dim",
                     default=5,
                     help="dimension of latent states/factor")
flags.DEFINE_multi_integer("non_mask_ar",
                     default=[1],
                     help="non-masked coeff of latent AR model")
flags.DEFINE_integer("id",
                     default=0,
                     help="id for setting seeds")
flags.DEFINE_float("learning_rate",
                     default=0.05,
                     help="learning rate")
flags.DEFINE_float("lambda_init",
                     default=1,
                     help="initial scaling of latent state transition matrix")
flags.DEFINE_integer("train_steps",
                     default=20,
                     help="number of GD optimization steps")
flags.DEFINE_string('output_dir',
                    default=os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "dfm_results_garch"),
                    help="Directory to save the results.")
flags.DEFINE_integer("d_agg_freq",
                     default=6,
                     help="number of daily days that are aggregated")
flags.DEFINE_integer("days_month",
                     default=24,
                     help="number of days (interpolated) per month")
flags.DEFINE_integer("q_lags",
                     default=12,
                     help="aggregation window quarterly")
flags.DEFINE_integer("m_lags",
                     default=4,
                     help="aggregation window monthly")
flags.DEFINE_integer("d_lags",
                     default=1,
                     help="aggregation window daily")
flags.DEFINE_string('model',
                    default='stock',
                    help="observation model type.")
flags.DEFINE_string('estimation',
                    default = 'fixed',
                    help = 'type of training (expanding, fixed or window')
flags.DEFINE_string('training_end',
                    default = '2007',
                    help = 'end of trainin perior (2007 or 2011)')
flags.DEFINE_bool('contractive',
                  default = True,
                  help = 'whether state transition dynamics should be contractive')


FLAGS = flags.FLAGS
FLAGS(sys.argv)
np.random.seed(FLAGS.id)

# save (command line) flags to file
key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
s = '\n'.join(f.serialize() for f in key_flags)
print('specified flags:\n{}'.format(s))
path = os.path.join(FLAGS.output_dir,
                    '{}__{}__{}__{}_{}__{}'.format(
                      FLAGS.model, FLAGS.latent_dim, FLAGS.d_agg_freq, FLAGS.non_mask_ar,
                      FLAGS.estimation, FLAGS.training_end))
if not os.path.exists(path):
  os.makedirs(path)
flag_file = open(os.path.join(path, 'flags.txt'), "w")
flag_file.write(s)
flag_file.close()



path_folder = Path(os.path.dirname(os.path.abspath(os.getcwd())))
data_path = os.path.join(path_folder, "", "DATE PRE-PROCESING")
#preproc_data = pd.read_csv(os.path.join(data_path, 'output_data_nosa.csv'),
#                             parse_dates = ["Date"]).set_index("Date")
#data = preproc_data.loc["1990-01-01":"2020-12-31"]
# Sample dates
startSampleDate = pd.to_datetime('1990-01-01')
endSampleDate   = pd.to_datetime('2021-07-31')

d_agg_freq = FLAGS.d_agg_freq


def interp1d2(y0, y1, n):
  x = np.linspace(0, 1, n + 2)
  y = y0 * (1 - x) + y1 * x
  return y[1:-1]


def daily_begginingInterp(data, start = None, end = None, length = None, endOfMonth = True):
  if isinstance(data, pd.DataFrame):
    data = data.squeeze()

  start_ = data.index[0] if (start is None) else start
  end_ = data.index[-1] if (end is None) else end

  assert ~(length is None) and (length > 0), "Choose a valid interpolation length"

  # start_ -= pd.offsets.MonthBegin(1)
  freqrange = pd.date_range(start = start_, end = end_, freq = 'MS')
  # if endOfMonth:
  #    freqrange = pd.date_range(start=start_, end=end_, freq='MS')
  # else:
  #    freqrange = pd.date_range(start=start_, end=end_, freq='MS')

  data_interp = np.full(length * len(freqrange), np.nan)
  # id_i = np.full(length * len(freqrange), np.nan)
  dates_interp = None

  # Initial case
  data_i0 = data.loc[:freqrange[1]]
  L = len(data_i0)
  data_interp[(length - L + 2):length] = data_i0.iloc[1:-1]
  data_interp[0:(length - L + 2)] = data_i0.iloc[0]
  data_f_last = data_i0[-2]
  if endOfMonth:
    dates_interp = pd.date_range(
      freqrange[1] - pd.DateOffset(days = length), freqrange[1] - pd.DateOffset(days = 1)
    )
  else:
    dates_interp = pd.date_range(
      freqrange[0], freqrange[0] + pd.DateOffset(days = length - 1)
    )

  # Full cases
  p = length
  for j in range(2, len(freqrange)):
    data_ij = data.loc[freqrange[j - 1]:freqrange[j]]
    L = len(data_ij)

    data_interp[(p + length - L + 2):(p + length)] = data_ij.iloc[1:-1]
    data_interp[p:(p + length - L + 2)] = interp1d2(
      data_f_last, data_ij[0], length - L + 2
    )
    if endOfMonth:
      dates_interp = dates_interp.append(pd.date_range(
        freqrange[j] - pd.DateOffset(days = length), freqrange[j] - pd.DateOffset(days = 1)
      ))
    else:
      dates_interp = dates_interp.append(pd.date_range(
        freqrange[j - 1], freqrange[j - 1] + pd.DateOffset(days = length - 1)
      ))

    # print(data_ij)
    # print(data_ij[0])

    # id_i[p:(p+length)] = j
    data_f_last = data_ij[-2]
    p += length

  # Last case
  data_ie = data.loc[freqrange[-1]:(freqrange[-1] + pd.tseries.offsets.MonthBegin())]
  L = len(data_ie)
  data_interp[(p + length - L + 1):] = data_ie.iloc[1:]
  data_interp[p:(p + length - L + 1)] = interp1d2(
    data_f_last, data_ie[0], length - L + 1
  )
  if endOfMonth:
    dates_interp = dates_interp.append(pd.date_range(
      freqrange[-1] + pd.tseries.offsets.MonthEnd() - pd.DateOffset(days = length - 1),
      freqrange[-1] + pd.tseries.offsets.MonthEnd()
    ))
  else:
    dates_interp = dates_interp.append(pd.date_range(
      freqrange[-1], freqrange[-1] + pd.DateOffset(days = length - 1)
    ))

  # data_interp[0:(length-L)] = interp1d2(d_data_pre[0], d_data_pre[1], 24-L)

  data_interp = pd.DataFrame(data = data_interp, index = dates_interp)

  return data_interp


def normalize_train_test(train, test):
  m_train = train.mean()
  s_train = (train - train.mean()).std()
  return (train - m_train) / s_train, (test - m_train) / s_train



# Non-filled data
preproc_data = pd.read_csv(
    os.path.join(data_path, 'fullsample_output_data.csv'),
    parse_dates = ["Date"]
).set_index("Date")
data = preproc_data.loc[startSampleDate:endSampleDate]

d_data = data[['D1']].dropna()
m_data = data[['M1', 'M4', 'M5', 'M7', 'M11', 'M12', 'M14', 'M15']].dropna()
GDP_data = data[['Q1']].dropna()

# Use squared returns as proxy for volatility
# d_data = d_data ** 2

# Interpolate non-filled daily data
d_data_interp = daily_begginingInterp(d_data, length=FLAGS.days_month, endOfMonth=False)
d_data_interp.columns = ['D1']

m_data_prefill = preproc_data[['M1', 'M4', 'M5', 'M7', 'M11', 'M12', 'M14', 'M15']].loc[
    (startSampleDate - pd.offsets.MonthBegin()):endSampleDate
].dropna()
GDP_data_prefill = preproc_data[['Q1']].loc[
    (startSampleDate - pd.offsets.QuarterBegin()):endSampleDate
].dropna()

# Interpolate non-filled daily data to month end
d_data_fill_interp = daily_begginingInterp(d_data, length=FLAGS.days_month, endOfMonth=True)
d_data_fill_interp.columns = ['D1']

# Shift monthly, quarterly data to solar end-of-month date
m_data_prefill.index = m_data_prefill.index + pd.offsets.MonthEnd(0)
GDP_data_prefill.index = GDP_data_prefill.index + pd.offsets.QuarterEnd(0)

# Re-align filled data to daily indexes, join dataframe
pre_index = pd.date_range(
    start=startSampleDate - pd.offsets.YearBegin(), end=endSampleDate, freq='D'
)
pre_data_interp = pd.DataFrame(
    data=np.full((len(pre_index), 1), np.nan), index=pre_index
)

m_data_interp = pre_data_interp.join(m_data_prefill).ffill()
Y_data_interp = pre_data_interp.join(GDP_data_prefill).ffill()

# Make a new dataframe for filled explanatory variables
md_fill_data = d_data_fill_interp.join(m_data_interp.iloc[:,1:])#.join(Y_data_interp.iloc[:,1:])

# Shift target data
GDP_fill_data  = GDP_data
GDP_fill_data.index = GDP_fill_data.index + pd.offsets.QuarterEnd(0)



def getSmallDatasets(trainTestSplitDate = None):
  assert not trainTestSplitDate is None, "Choose a slicing data for train/test datasets"

  # trainTestSplitDate = pd.to_datetime('2007-12-31')
  trainTestSplitDate = pd.to_datetime(trainTestSplitDate)

  # Split training and testing sample
  GDP_data_train = GDP_data.loc[startSampleDate:trainTestSplitDate]
  m_data_train = m_data.loc[startSampleDate:trainTestSplitDate]
  d_data_train = d_data_interp.loc[startSampleDate:trainTestSplitDate]
  GDP_fill_data_train = GDP_fill_data.loc[startSampleDate:trainTestSplitDate]
  md_fill_data_train = md_fill_data.loc[startSampleDate:trainTestSplitDate]

  GDP_data_test = GDP_data.loc[(trainTestSplitDate + pd.offsets.Day()):endSampleDate]
  m_data_test = m_data.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1)):endSampleDate]
  d_data_test = d_data_interp.loc[(trainTestSplitDate - pd.offsets.MonthBegin(1) + pd.offsets.Day(23)):endSampleDate]
  GDP_fill_data_test = GDP_fill_data.loc[(trainTestSplitDate + pd.offsets.Day()):endSampleDate]
  md_fill_data_test = md_fill_data.loc[(trainTestSplitDate):endSampleDate]

  # Normalize
  GDP_data_train, GDP_data_test = normalize_train_test(GDP_data_train, GDP_data_test)
  m_data_train, m_data_test = normalize_train_test(m_data_train, m_data_test)
  d_data_train, d_data_test = normalize_train_test(d_data_train, d_data_test)

  GDP_fill_data_train, GDP_fill_data_test = normalize_train_test(GDP_fill_data_train, GDP_fill_data_test)
  md_fill_data_train, md_fill_data_test = normalize_train_test(md_fill_data_train, md_fill_data_test)

  # MIDAS data format
  GDP_data_midas = np.vstack((
    GDP_data_train.to_numpy(),
    GDP_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
  ))
  md_data_midas = (
    tuple(
      np.vstack((
        m_data_train[s].to_numpy()[:, None],
        m_data_test[s].loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()[:, None]
      )) for s in m_data_train.columns
    ) +
    (np.vstack((
      d_data_train.to_numpy(),
      d_data_test.loc[(trainTestSplitDate + pd.offsets.Day()):].to_numpy()
    )),)
  )

  dataset = {
    'GDP_data_train': GDP_data_train,
    'GDP_data_test': GDP_data_test,
    'm_data_train': m_data_train,
    'm_data_test': m_data_test,
    'd_data_train': d_data_train,
    'd_data_test': d_data_test,
    'GDP_fill_data_train': GDP_fill_data_train,
    'GDP_fill_data_test': GDP_fill_data_test,
    'md_fill_data_train': md_fill_data_train,
    'md_fill_data_test': md_fill_data_test,
    'GDP_data_midas': GDP_data_midas,
    'md_data_midas': md_data_midas,
  }

  return dataset


if FLAGS.training_end == '2007':
  dataset2007 = getSmallDatasets('2007-12-31')
  d_data_train = dataset2007['d_data_train']
  d_data_test = dataset2007['d_data_test'].iloc[1:]
  m_data_train = dataset2007['m_data_train']
  m_data_test = dataset2007['m_data_test'].iloc[1:]
  q_data_train = dataset2007['GDP_data_train']
  q_data_test = dataset2007['GDP_data_test']
elif FLAGS.training_end == '2011':
  dataset2011 = getSmallDatasets('2011-12-31')
  d_data_train = dataset2011['d_data_train']
  d_data_test = dataset2011['d_data_test'].iloc[1:]
  m_data_train = dataset2011['m_data_train']
  m_data_test = dataset2011['m_data_test'].iloc[1:]
  q_data_train = dataset2011['GDP_data_train']
  q_data_test = dataset2011['GDP_data_test']
q_data = pd.concat([q_data_train, q_data_test])
m_data = pd.concat([m_data_train, m_data_test])
d_data = pd.concat([d_data_train, d_data_test])


##
#DFM data handling
##
if d_agg_freq>1:
  mean_d_data_train = d_data_train.groupby(np.arange(len(d_data_train))//d_agg_freq).mean()
  d_data_train = pd.DataFrame(mean_d_data_train).set_index(d_data_train[d_agg_freq-1::d_agg_freq].index)
  mean_d_data_test = d_data_test.groupby(np.arange(len(d_data_test))//d_agg_freq).mean()
  d_data_test = pd.DataFrame(mean_d_data_test).set_index(d_data_test[d_agg_freq-1::d_agg_freq].index)





#q_data = pd.DataFrame(Y)
#d_data = pd.DataFrame(d_data)
#q_data_train = pd.DataFrame(Y_train)
#q_data_test= pd.DataFrame(Y_test)

q_symbols = q_data.columns.to_list()
m_symbols = m_data.columns.to_list()
d_symbols = d_data.columns.to_list()

q_dates = q_data.index.to_list()
m_dates = m_data.index.to_list()
d_dates = d_data.index.to_list()

q_dates_train = q_data_train.index.to_list()
m_dates_train = m_data_train.index.to_list()
d_dates_train = d_data_train.index.to_list()

q_dates_test = q_data_test.index.to_list()
m_dates_test = m_data_test.index.to_list()
d_dates_test = d_data_test.index.to_list()

q_dims = q_data.shape[-1]
m_dims = m_data.shape[-1]
d_dims = d_data.shape[-1]




data_train = pd.merge(q_data_train, m_data_train, left_index = True, right_index = True, how='outer')
data_train = pd.merge(data_train, d_data_train, left_index = True, right_index = True, how='outer')
data_train = data_train.fillna(method='ffill')
data_train = data_train.fillna(0)


data_test = pd.merge(q_data_test, m_data_test, left_index = True, right_index = True, how='outer')
data_test = pd.merge(data_test, d_data_test, left_index = True, right_index = True, how='outer')
data_test= data_test.fillna(method='ffill')
data_test = data_test.fillna(0)


def map_timestamp_to_model(t):
  w_dates = []
  if t in q_dates and t not in w_dates:
    return 2  # quarterly, monthly and daily observations
  elif t in m_dates and t not in w_dates:
    return 1  # monthly and daily observations
  elif t in d_dates and t not in w_dates:
     return 0  # only daily observations
  else:
    assert(1==2)

  # elif t in q_dates and t  in w_dates:
  #   return 5  # quarterly, monthly and daily observations
  # elif t in m_dates and t  in w_dates:
  #   return 4  # monthly and daily observations
  # elif t in d_dates and t  in w_dates:
  #    return 3  # only daily observations
  #
  # else:
  #   return -1

linear_gaussian_model = QMWDDFM(non_mask_ar = FLAGS.non_mask_ar,
  q_process_dim = q_dims, m_process_dim = m_dims, w_process_dim = 0, d_process_dim = d_dims,
  latent_process_dim = FLAGS.latent_dim, map_timestamp_to_model = map_timestamp_to_model,
  dtype = tf.float64, q_lags = FLAGS.q_lags, m_lags = FLAGS.m_lags, d_lags = FLAGS.d_lags,
  model = FLAGS.model, lambda_init = FLAGS.lambda_init, contractive = FLAGS.contractive)

print(linear_gaussian_model._params)
pd.DataFrame((tf.linalg.eigh(linear_gaussian_model._transition_matrix_fn(0).to_dense())[0].numpy())).to_csv(
  os.path.join(path, 'eigenvalues_train0.csv'))
pd.DataFrame((linear_gaussian_model._transition_matrix_fn(0).to_dense()).numpy()).to_csv(
  os.path.join(path, 'A_train0.csv'))

linear_gaussian_model.fit(y_train = data_train, train_dates = data_train.index, learning_rate=FLAGS.learning_rate,
                          steps=FLAGS.train_steps)

print(linear_gaussian_model._params)




#train predictions
#in-sample predicions
train_predict_dates = data_train.index
w_dims = 0
predictions_train = []

factor_predictions_train, observation_predictions_train = linear_gaussian_model.predict_multiple(
  0, y_test = data_train[:-FLAGS.horizon], predict_dates = data_train.index,
  horizon = FLAGS.horizon)

for h in range(FLAGS.horizon):
  factor_predictions_train[h].to_csv(os.path.join(path, 'train_factor_predictions_{}.csv'.format(h)))
  observation_predictions_train[h].to_csv(os.path.join(path, 'train_predictions_{}.csv'.format(h)))



# for horizon in FLAGS.horizon:
#   print(horizon)
#   kalman_filter_predictions_train = linear_gaussian_model.predict_graph(y_test = data_train[:-horizon], predict_dates = train_predict_dates,
#                                                                   horizon = horizon)
#   q_predict_dates_train = q_dates_train[1+int(horizon//(3*FLAGS.days_month/d_agg_freq)):]
#   m_predict_dates_train = m_dates_train[int(horizon//(FLAGS.days_month/d_agg_freq)):]
#   d_predict_dates_train = d_dates_train[horizon:]
#   q_predictions_train = kalman_filter_predictions_train.droplevel(0).loc[
#     q_predict_dates_train][[i for i in range(q_dims)]].rename(
#     columns={i:sym for i, sym in zip(range(q_dims), q_symbols)})
#   m_predictions_train = kalman_filter_predictions_train.droplevel(0).loc[
#     m_predict_dates_train][[i for i in range(q_dims,q_dims+m_dims)]].rename(
#     columns={i:sym for i, sym in zip(range(q_dims,q_dims+m_dims), m_symbols)})
#   d_predictions_train = kalman_filter_predictions_train.droplevel(0).loc[d_predict_dates_train][[i for i in range(
#     q_dims+m_dims+w_dims,q_dims+m_dims+w_dims+d_dims)]].rename(
#     columns={i:sym for i, sym in zip(range(q_dims+m_dims+w_dims,q_dims+m_dims+w_dims+d_dims), d_symbols)})
#
#   q_prediction_error_train = ((q_predictions_train - q_data.loc[q_predict_dates_train])**2).sum()
#   m_prediction_error_train = ((m_predictions_train - m_data.loc[m_predict_dates_train])**2).sum()
#   d_prediction_error_train = ((d_predictions_train - d_data.loc[d_predict_dates_train])**2).sum()
#
#   print(q_prediction_error_train)
#   print(m_prediction_error_train)
#   print(d_prediction_error_train)
#
#   q_prediction_error_train.to_csv(os.path.join(path, 'q_prediction_error_train_{}.csv'.format(
#     horizon)))
#   m_prediction_error_train.to_csv(os.path.join(path, 'm_prediction_error_train_{}.csv'.format(
#     horizon)))
#   d_prediction_error_train.to_csv(os.path.join(path, 'd_prediction_error_train_{}.csv'.format(
#     horizon)))
#
#
#   df_q_train = pd.concat([q_predictions_train.rename(columns=lambda s: s+'_train_pred'),
#                           q_data_train.rename(columns=lambda s: s+'_train')], 1)
#   df_q_train.to_csv(os.path.join(path, 'q_train_{}.csv'.format(horizon)))
#   df_m_train = pd.concat([m_predictions_train.rename(columns=lambda s: s+'_train_pred'),
#                           m_data_train.rename(columns=lambda s: s+'_train')], 1)
#   df_m_train.to_csv(os.path.join(path, 'm_train_{}.csv'.format(horizon)))
#
#
#   for q_symbol in q_symbols:
#     fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
#     df_m_train = pd.concat([pd.DataFrame(q_predictions_train[q_symbol]).rename(columns = lambda s: s + '_train_pred'),
#                             pd.DataFrame(q_data_train[q_symbol]).rename(columns = lambda s: s + '_train')], 1)
#     df_m_train.plot(ax = axes)
#     plt.title(q_symbol + ' nowcasting {} steps'.format(horizon))
#     plt.legend()
#     plt.savefig(
#       os.path.join(path, q_symbol +'_forecast_{}.png'.format(horizon)))
#     plt.close()
#
#   for m_symbol in m_symbols:
#     fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
#     df_m_train = pd.concat([pd.DataFrame(m_predictions_train[m_symbol]).rename(columns = lambda s: s + '_train_pred'),
#                             pd.DataFrame(m_data_train[m_symbol]).rename(columns = lambda s: s + '_train')], 1)
#     df_m_train.plot(ax = axes)
#     plt.title(m_symbol + ' nowcasting {} steps'.format(horizon))
#     plt.legend()
#     plt.savefig(
#       os.path.join(path, m_symbol +'_forecast_{}.png'.format(horizon)))
#     plt.close()
#
#
#     predictions_train.append(kalman_filter_predictions_train)
#
# predictions_train = pd.concat(predictions_train, 0)
# #predictions_train.to_csv(os.path.join(path, 'predictions_train.csv'))

pd.DataFrame((tf.linalg.eigh(linear_gaussian_model._transition_matrix_fn(0).to_dense())[0].numpy())).to_csv(
  os.path.join(path, 'eigenvalues_train.csv'))
pd.DataFrame((linear_gaussian_model._transition_matrix_fn(0).to_dense()).numpy()).to_csv(
  os.path.join(path, 'A_train.csv'))

predictions_test = []
predictions_test_horizon = {h: [] for h in FLAGS.horizon}
#Test set predictions by updating the static parameters online
last_pred_date = {h: kalman_filter_predictions_train.iloc[-1].name[0] for h in FLAGS.horizon}

update_freq = 3*FLAGS.days_month//d_agg_freq
for s in np.arange(0, data_test.shape[0]+update_freq, update_freq):

  pd.DataFrame((tf.linalg.eigh(linear_gaussian_model._transition_matrix_fn(0).to_dense())[0].numpy())).to_csv(
    os.path.join(path, 'eigenvalues_test_{}.csv'.format(s)))

  if FLAGS.estimation in ['fixed','expanding']:
    data_s = pd.concat([data_train, data_test.iloc[:min(s, data_test.shape[0])]], 0)
  elif FLAGS.estimation in ['window']:
    data_s = pd.concat([data_train, data_test.iloc[:min(s, data_test.shape[0])]], 0)
    data_s = data_s.iloc[-data_train.shape[0]:]

  #update by running SGD:
  if s % (update_freq) == 0:

    for horizon in FLAGS.horizon:
      print(horizon)
      if s+horizon>data_test.shape[0]:
        continue
      predict_dates = pd.concat([data_s, data_test.iloc[min(s, data_test.shape[0]):min(s+horizon, data_test.shape[0])]], 0).index
      kalman_filter_predictions_test = linear_gaussian_model.predict_graph(y_test = data_s, predict_dates = predict_dates, horizon = horizon)

      predictions_test.append(kalman_filter_predictions_test[last_pred_date[horizon]:].iloc[1:])
      predictions_test_horizon[horizon].append(kalman_filter_predictions_test[last_pred_date[horizon]:].iloc[1:])
      last_pred_date[horizon] = kalman_filter_predictions_test.iloc[-1].name[0]

    #update the static parameters after making the predictions by running a few SGD iterations on the full data
    #(as available at that time)
    if not FLAGS.estimation == 'fixed':
      linear_gaussian_model.fit(y_train = data_s, train_dates = data_s.index, learning_rate = FLAGS.learning_rate,
                            steps = 50)


predictions_test = pd.concat(predictions_test, 0)
predictions_test.to_csv(os.path.join(path, 'predictions_test.csv'))

for horizon in FLAGS.horizon:

  q_predictions_test = pd.concat(predictions_test_horizon[horizon], 0).droplevel(0)[[i for i in range(q_dims)]].rename(columns = {i: sym for i, sym in zip(range(q_dims), q_symbols)})
  m_predictions_test = pd.concat(predictions_test_horizon[horizon], 0).droplevel(0)[[i for i in range(q_dims, q_dims + m_dims)]].rename(columns = {i: sym for i, sym in zip(range(q_dims, q_dims + m_dims), m_symbols)})
  for m_symbol in m_symbols:
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
    df_m_test = pd.concat([pd.DataFrame(m_predictions_test[m_symbol]).rename(columns = lambda s: s + '_test_pred'),pd.DataFrame(m_data_test[m_symbol]).rename(columns = lambda s: s + '_test')], 1).loc[m_dates_test]
    df_m_test.plot(ax = axes)
    plt.title(m_symbol + ' nowcasting {} steps'.format(horizon))
    plt.legend()
    plt.savefig(
      os.path.join(path, m_symbol +'_forecast_test_{}.png'.format(horizon)))
    plt.close()
    df_m_test.to_csv(os.path.join(path, 'm_test_{}.csv'.format(horizon)))

  for q_symbol in q_symbols:
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
    df_q_test = pd.concat([pd.DataFrame(q_predictions_test[q_symbol]).rename(columns = lambda s: s + '_test_pred'),pd.DataFrame(q_data_test[q_symbol]).rename(columns = lambda s: s + '_test')], 1).loc[q_dates_test]
    df_q_test.plot(ax = axes)
    plt.title(q_symbol + ' nowcasting {} steps'.format(horizon))
    plt.legend()
    plt.savefig(
      os.path.join(path, q_symbol +'_forecast_test{}.png'.format(horizon)))
    plt.close()
    df_q_test.to_csv(os.path.join(path, 'q_test_{}.csv'.format(horizon)))

