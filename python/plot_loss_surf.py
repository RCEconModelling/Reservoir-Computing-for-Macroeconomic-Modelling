import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import interp2d
from scipy.ndimage import minimum_filter
from tqdm import tqdm
#import multiprocessing
#from joblib import Parallel, delayed
#import plotly.express as px

# Set up plot axes
plt.rc('axes', axisbelow=True)


def plot_loss_surf(loss_fn, grad_fn, param, scale = 1, scale_ub = None, scale_lb = None,
                    delta = None, nu = None, M = 100, normalize = True, seed=None, figsize=None):
  """args:
  loss_fn: function for visualisation
  param: parameter vector around which to perturb loss surface
  deta, nu: directions (if None: they are drawn randomly on the unit sphere
     and normalized relative to the norm of param if normailize is True)
  M: grid size
  """

  # Set seed
  if not seed is None:
      rng = np.random.default_rng(seed)
  else:
      rng = np.random.default_rng(12345)

  # Parallelize using 'joblib' on multicore machine
  #num_cores = multiprocessing.cpu_count()

  scale_ub = +scale if (scale_ub is None) else scale_ub
  scale_lb = -scale if (scale_lb is None) else scale_lb

  linsp = np.linspace(scale_lb, scale_ub, M)
  #linsp = np.linspace(0, scale, M)
  Z = np.full((M, M), np.nan)
  G = np.full((M, M), np.nan)

  if delta is None:
    delta = rng.standard_normal(size=param.shape)
    delta /= np.linalg.norm(delta)
    if normalize:
      delta = delta * np.linalg.norm(param)#

  if nu is None:
    nu = rng.standard_normal(size=param.shape)
    nu /= np.linalg.norm(nu)
    if normalize:
      nu = nu * np.linalg.norm(param)

  #delta = param - nu

  #param = param + 0.034*delta - 0.015*nu
  #param0 = np.zeros(delta.shape)
  param0 = param

  for n in tqdm (range(len(linsp)), desc="Evaluating loss..."):
      theta_opt_n = param0 + linsp[n]*delta

      #def eval_loss(m):
      #  return loss_fn((theta_opt_n + linsp[m]*nu))
      #def eval_grad(m):
      #  return np.sqrt(np.sum((grad_fn((theta_opt_n + linsp[m]*nu))**2)))
      #Z[n,:] = Parallel(n_jobs=4)(delayed(eval_loss)(m) for m in range(len(linsp)))
      #G[n,:] = Parallel(n_jobs=4)(delayed(eval_grad)(m) for m in range(len(linsp)))

      for m in range(len(linsp)):
          Z[n,m] = loss_fn((theta_opt_n + linsp[m]*nu))
          G[n,m] = np.linalg.norm(grad_fn((theta_opt_n + linsp[m]*nu)))

  #grad_origin = np.linalg.norm((grad_fn(param)))

  X, Y = np.meshgrid(linsp, linsp)
  #xi = np.linspace(linsp.min(), linsp.max(), (M*2))
  #yi = np.linspace(linsp.min(), linsp.max(), (M*2))
  #zi = interp2d(linsp, linsp, np.log(Z), kind="cubic")
  #gi = interp2d(linsp, linsp, np.log(G), kind="cubic")
  #xig, yig = np.meshgrid(xi, yi)

  # Check for local minima
  #locminX, locminY = np.where((Z == minimum_filter(Z, size=5, mode='constant', cval=0.0)))

  #print(np.sqrt(np.sum((grad_fn((param))**2))))

  #fig = px.imshow(np.log(Z))
  #fig.show()

  figsize = (5,5) if figsize is None else figsize

  # Plot
  fig = plt.figure(figsize=figsize)

  ax1 = fig.add_subplot(1,2,1)
  #ax1.axes.contourf(X, Y, np.log(Z), levels=30, cmap=plt.cm.bone.reversed())
  pcm1 = ax1.axes.pcolormesh(X, Y, (Z), cmap=plt.cm.nipy_spectral, shading='auto', norm=colors.LogNorm())
  plt.colorbar(pcm1)
  #ax1.axes.pcolormesh(xig, yig, zi(xi, yi), cmap=plt.cm.coolwarm, shading='auto')
  #ax1.axes.scatter(linsp[locminY], linsp[locminX], color='white', marker='.', s=10)
  ax1.axes.scatter(0, 0, color='white', marker='.')
  #ax1.axes.scatter(0, 1, color='white', marker='.')
  #x1.axes.scatter(1, 0, color='white', marker='.')
  #ax1.xaxis.get_data_interval()
  #ax1.yaxis.get_data_interval()
  ax1.title.set_text(r"$\mathcal{L}$")

  #fig2, _ = plt.subplots(figsize=(5, 5))
  ax2 = fig.add_subplot(1,2,2)
  #ax2=axes[1]
  #ax2.axes.plot_surface(X, Y, np.log(Z), rcount=100, cmap=plt.cm.coolwarm)
  #ax2.axes.plot_surface(xig, yig, zi(xi, yi), rcount=150, cmap=plt.cm.coolwarm)
  #ax2.axes.plot_trisurf(xig.reshape(-1), yig.reshape(-1), zi(xi, yi).reshape(-1), linewidth=0, antialiased=True,  cmap=plt.cm.coolwarm)
  #ax2.axes.scatter(0, 0, np.log((loss_fn(param))), color='black', marker='o')
  
  pcm2 = ax2.axes.pcolormesh(X, Y, (G), cmap=plt.cm.nipy_spectral, shading='auto', norm=colors.LogNorm())
  plt.colorbar(pcm2)
  #ax2.axes.pcolormesh(xig, yig, gi(xi, yi), cmap=plt.cm.coolwarm, shading='auto')
  #ax2.axes.scatter(linsp[locminY], linsp[locminX], color='white', marker='.', s=10)
  ax2.axes.scatter(0, 0, color='white', marker='.')
  #ax2.axes.scatter(0, 1, color='white', marker='.')
  #ax2.axes.scatter(1, 0, color='white', marker='.')
  ax2.title.set_text(r"$||\nabla\mathcal{L}||$")

  #ax3 = fig.add_subplot(1,3,3)
  #ax3.axes.pcolormesh(X, Y, np.log(G), cmap=plt.cm.coolwarm)

  plt.show()

  return delta, nu, X, Y, Z