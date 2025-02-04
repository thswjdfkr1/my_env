import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
from copy import deepcopy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
#%matplotlib inline

def lm_stat(model, X, y, alternative="two_sided", variables=None, digits=3):
  """
  @params \n
  model : sklearn.linearmodel.LinearRegression().fit(X, y) \n
  X : independent variables of model \n
  y : dependent variable of model \n
  alternative : a character string specifying the alternative hypothesis, must be one of "two_sided" (default), "greater" or "less". You can specify just the initial letter.
  """
  params = np.append(model.intercept_, model.coef_)
  pred = model.predict(X)
  N = X.shape[0]
  newX = pd.DataFrame({'Constant': np.ones(len(X))}).join(pd.DataFrame(X))
  MSE = (sum((y-pred)**2)) / (len(newX) - len(newX.columns))

  var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
  sd_b = np.sqrt(var_b)
  ts_b = params / sd_b

  if (alternative == 'two_sided') or (alternative == 't'):
    p_val = [2*(1-stats.t.cdf(np.abs(ts), N-2)) for ts in ts_b]
    p_val_name = 'Pr(>|t_val|)'
  elif (alternative == 'greater') or (alternative == 'g'):
    p_val = [(1-stats.t.cdf(ts, N-2)) for ts in ts_b]
    p_val_name = 'Pr(>t_val)'
  elif (alternative == 'less') or (alternative == 'l'):
    p_val = [(stats.t.cdf(ts, N-2)) for ts in ts_b]
    p_val_name = 'Pr(<t_val)'
  else:
    print("ERROR: \nChoose the SPECIFIC parameter for 'alternative'")

  sd_b = np.round(sd_b, digits)
  ts_b = np.round(ts_b ,digits)

  p_val = np.round(p_val, digits)
  params = np.round(params, digits)

  df = pd.DataFrame()
  df['coef'], df['se'], df['t_val'], df[p_val_name] = [params, sd_b, ts_b, p_val]

  if variables == None:
    variables = ['Beta'+str(i+1) for i in range(X.shape[1])]

  variables.insert(0, 'Intercept')
  df.index = variables
  return df

def lm_r2(model, X, y, adjust=True, digits=3):
  N, p = X.shape
  pred = model.predict(X)
  ssr = sum((y-pred)**2)
  sst = sum((y-np.mean(y))**2)

  r2 = 1 - (float(ssr)) / sst

  if adjust:
    adj_r2 = 1 - (1-r2) * (N-1) / (N-p-1)
    return round(float(adj_r2), digits)
  else:
    return round(float(r2), digits)

def stepwise(X, y, model_type='linear', thred=0.05, variables=None, logit_method='bfgs', disp=0):
  """
  @params \n
  X : independent variables \n
  y : dependent variable \n
  model_type : 'linear' (for Linear regression by default) or 'logit' (for Logistic regression)
  thred : p-value's threshold for stepwise selection. (default) 0.05
  variables : (list) column names of X
  """
  warnings.filterwarnings("ignore")
  if variables == None:
    X = pd.DataFrame(X)
    variables = ['V'+str(v) for v in range(X.shape[1])]
    X.columns = variables
  else:
    X = pd.DataFrame(X, columns=variables)

  features = deepcopy(variables)
  selected = []
  
  #sv_per_step = []
  #adj_r2 = []
  #steps = []
  #step = 0

  while len(features) > 0:
    remained = list(set(features) - set(selected))
    pval = pd.Series(index=remained)
    for col in remained:
      x = X[selected + [col]]
      x = sm.add_constant(x)

      if model_type == 'linear':
        model = sm.OLS(y, x).fit(disp=disp)
      elif model_type == 'logit':
        model = sm.Logit(y, x).fit(method=logit_method, disp=disp)

      pval[col] = model.pvalues[col]

    min_pval = pval.min()
    if min_pval < thred:
      selected.append(pval.idxmin())
      
      while len(selected) > 0:
        selected_X = X[selected]
        selected_X = sm.add_constant(selected_X)

        if model_type == 'linear':
          selected_pval = sm.OLS(y, selected_X).fit(disp=disp).pvalues[1:]
        elif model_type == 'logit':
          selected_pval = sm.Logit(y, selected_X).fit(method=logit_method, disp=disp).pvalues[1:]

        max_pval = selected_pval.max()

        if max_pval >= thred:
          remove_variable = selected_pval.idxmax()
          selected.remove(remove_variable)
        else:
          break

      #step += 1
      #steps.append(step)
      #adj_r2_val = sm.OLS(y_train, sm.add_constant(X_train[selected])).fit(disp=0).rsquared_adj
      #adj_r2.append(adj_r2_val)
      #sv_per_step.append(selected.copy())
    else:
      break
  return selected

def viz_boundary(model, X, y, variables=None,
                 margin=1, h=300, color_list=['r', 'g', 'b'],
                 figsize=(5, 5), alpha=0.1, markersize=2,
                 xlab='X1', ylab='X2', fontsize=10, rotation=0):
  """
  @params \n
  model : classifier in sklearn
  X : (numpy.array) 2-dimensional independent variables of model \n
  y : dependent variable of model \n
  """
  if type(X) != np.ndarray:
    X = np.array(X)
    y = np.array(y)
  if X.shape[1] == 2:
    mn_x1 = int(X[:, 0].min()) - margin
    mx_x1 = int(X[:, 0].max()) + margin

    mn_x2 = int(X[:, 1].min()) - margin
    mx_x2 = int(X[:, 1].max()) + margin

    x1s = np.linspace(mn_x1, mx_x1, h)
    x2s = np.linspace(mn_x2, mx_x2, h)

    x1, x2 = np.meshgrid(x1s, x2s)
    x_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = model.predict(x_new).reshape(x1.shape)
    axes = [mn_x1, mx_x1, mn_x2, mx_x2]

    unique_y = list(np.unique(y))
    y_col = color_list[:len(unique_y)]  
    custom_cmap = ListedColormap(y_col)

    plt.figure(figsize=figsize)
    plt.contourf(x1, x2, y_pred, cmap=custom_cmap, alpha=alpha)

    for i, yi in enumerate(unique_y):
      ci = y_col[i] + 'o'
      plt.plot(X[:,0][y==yi], X[:,1][y==yi], ci, markersize=markersize)
    plt.axis(axes)
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize, rotation=rotation)

    plt.show()
  else:
    print("X should be 2-dimensional numpy.array")
    
if __name__ == '__main__':
  from dslearn import multi_stat

  # Load Dataset
  from sklearn.datasets import load_diabetes
  X, y = load_diabetes(return_X_y=True)

  # Statistical Test (Beta1 = 0)
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression().fit(X, y)
  print(multi_stat.lm_stat(model=lm, X=X, y=y))

  # get R2 or adj-R2
  print("R2 =", multi_stat.lm_r2(model=lm, X=X, y=y, adjust=False))
  print("adj-R2 =", multi_stat.lm_r2(model=lm, X=X, y=y, adjust=True))

  # Feature Selection (with Linear Regression)
  print("Selected variables:", stepwise(X=X, y=y))

  # Feature Selection (with Logistic Regression)
  print("Selected variables:", stepwise(X=X, y=y, model_type='logit'))
