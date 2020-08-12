# Standard modules
import numpy as np
import pandas as pd
import statistics as stats
import os.path as op

# Own modules
import nnet as nn
import strplots as plts
import galg as ga

### - Parameters

## -- Metaparameters
path = ''
figs_folder = 'figures'
seed = False

## -- Neural Network(s) parameters

t_size = 0.2  # test set size as a percentage of the whole set
neurons = [3, 4, 5]  # possible number of neurons per hidden layer
layers = [2, 4]  # possible number of hidden layers
st_leads = 17  # short-term leads of dependent variable
mt_leads = 34  # mid-term leads of dependent variable
lags = 7  # lags to include for each explanatory variable
terms = [st_leads, st_leads, mt_leads, mt_leads]

## -- Genetic Algorithm parameters

a1 = 1  # importance of the 4-month ahead inflation
b1 = 1  # importance of the 4-month ahead unemployment
a2 = 1  # importance of the 8-month ahead inflation
b2 = 1 # importance of the 8-month ahead unemployment
d_rate = .00012  # one week discount rate
n_breeders = 3  # number of breeders per generation
mut_perc = .07  # initial percentage of mutated genes
selic_options = np.arange(-3, 3, .1)  # possible interest rate choices
interval = 7 # how often the central bank chooses the interest rate, in weeks

### - Neural Network(s)

## -- Load data
data = pd.read_csv(path + 'weekly_data.csv')
data = data.iloc[::-1]
data = data.reset_index(drop=True)
data_orig = data.copy()

### - Strategy

## -- Running the regressions
if seed:
    np.random.seed(534)

# Scaling data
data = nn.scale(data)
data.to_csv('scaled_data.csv', index=False)

# Define regression procedure
def run_regressions(n_ests, plot=True):

    if n_ests == 1:


        # Splitting between training and testing sets
        X_train, y_train, X_test, y_test, X_orig, y_orig, indx = nn.split(data, lags, terms, t_size, random = True)

        # Run
        preds_in, preds_out, estimator, testing_metrics, training_metrics = nn.regression(X_train, y_train, X_test, y_test, neurons, layers)

        # Plotting resulsts
        if plot:
            plts.plot_nn_performance(preds_out, preds_in, indx, data_orig.dates, terms, y_orig, lags, figs_folder)

        if not op.isfile('X_orig.csv'):
            X_orig.to_csv('X_orig.csv', index=False)

        if not op.isfile('y_orig.csv'):
            y_orig.to_csv('y_orig.csv', index=False)

        estimators = [estimator]

    else:
        s_term = [st_leads, st_leads]
        m_term = [mt_leads, mt_leads]

        # Splitting between training and testing sets
        X_train_s, y_train_s, X_test_s, y_test_s, X_orig_s, y_orig_s, indx_s = nn.split(data, lags, s_term, t_size, random = False)
        X_train_m, y_train_m, X_test_m, y_test_m, X_orig_m, y_orig_m, indx_m = nn.split(data, lags, m_term, t_size, random = False)

        preds_in_s, preds_out_s, estimator_s, testm_s, trainm_s = nn.regression(X_train_s, y_train_s, X_test_s, y_test_s, neurons, layers)
        preds_in_m, preds_out_m, estimator_m, testm_m, trainm_m = nn.regression(X_train_m, y_train_m, X_test_m, y_test_m, neurons, layers)

        # Plotting results
        if plot:
            plts.plot_nn_performance(preds_out_s, preds_in_s, indx_s, data_orig.dates, s_term, y_orig_s, lags, figs_folder,'short_term')
            plts.plot_nn_performance(preds_out_m, preds_in_m, indx_m, data_orig.dates, s_term, y_orig_m, lags, figs_folder,'mid_term')

        testing_metrics = {'R2': stats.mean([testm_m['R2'], testm_s['R2']]),
                           'R2 Adj': stats.mean([testm_m['R2 Adj'], testm_s['R2 Adj']]),
                           'RMSE': stats.mean([testm_m['RMSE'], testm_s['RMSE']])
                           }

        training_metrics = {'R2': stats.mean([trainm_m['R2'], trainm_s['R2']]),
                           'R2 Adj': stats.mean([trainm_m['R2 Adj'], trainm_s['R2 Adj']]),
                           'RMSE': stats.mean([trainm_m['RMSE'], trainm_s['RMSE']]),
                           'CV Mean RMSE': stats.mean([trainm_m['CV Mean RMSE'], trainm_s['CV Mean RMSE']])
                           }

        estimators = [estimator_s, estimator_m]

    return estimators, testing_metrics, training_metrics

# Run regressions

#uni_estimator, uni_test_scores, uni_train_scores = run_regressions(1, True)
#sep_estimators, sep_test_scores, sep_train_scores = run_regressions(2, True)

#nn.save_estimators(sep_estimators, '')
#nn.save_estimators(uni_estimator, '')


## -- Optimization

X_orig = pd.read_csv('X_orig.csv')
y_orig = pd.read_csv('y_orig.csv')
weights = [a1, b1, a2, b2]
#estimators = [nn.get_estimator('st'), nn.get_estimator('mt')]
estimators = [nn.get_estimator('un')]

solutions, optimal_values, X = ga.run_optimization(X_orig, y_orig, estimators, weights, d_rate, n_breeders, terms, mut_perc, selic_options, lags, interval, runs = 2000)
ga.save_outcomes(solutions, optimal_values, X, prfx='')

# Plot original solution
#solutions, optimal_values, X = ga.get_outcomes('alt_')

selic_orig = X_orig['meta.selic']
scaler = nn.get_scaler()

selic_mean = scaler.mean_[10]
selic_var  = scaler.var_[10]
sols = np.array(solutions)*(selic_var**.5) + selic_mean
orig = np.array(selic_orig)*(selic_var**.5) + selic_mean
plts.plot_interest(sols, orig, data_orig, y_orig, lags, folder='figures')

preds1 = estimators[0].predict(X)
if len(estimators) == 2:
    preds2 = estimators[1].predict(X)
    preds = pd.concat([pd.DataFrame(preds1), pd.DataFrame(preds2)], axis=1)
else:
    preds = pd.DataFrame(preds1)
preds.columns = y_orig.columns
plts.plot_sol(preds, y_orig, data_orig, lags, folder='figures')

plts.plot_evolution(optimal_values, folder='figures')

# Plot MA alternative
sols_ma = pd.DataFrame(sols).rolling(window=12*lags).mean()
plts.plot_interest(sols_ma, orig, data_orig, y_orig, lags,folder='figures_ma')

X_ma = ga.get_mas(X, 12, lags)

preds1 = estimators[0].predict(X_ma)

if len(estimators) == 2:
    preds2 = estimators[1].predict(X_ma)
    preds = pd.concat([pd.DataFrame(preds1), pd.DataFrame(preds2)], axis=1)
else:
    preds = pd.DataFrame(preds1)
preds.columns = y_orig.columns
plts.plot_sol(preds, y_orig, data_orig, lags, folder='figures_ma')

l_preds = len(preds)
l_sols = len(sols_ma)
l_orig = len(orig)
l_y = len(y_orig)

sols_ma = sols_ma[(l_sols - l_preds):].reset_index(drop=True)
sols_ma.columns = ['selic']
pd.concat([preds, sols_ma], axis=1).to_csv('solutions.csv', index=False)

orig = pd.DataFrame(orig[(l_orig - l_preds):]).reset_index(drop=True)
orig.columns = ['selic']
y_orig = y_orig[(l_y - l_preds):].reset_index(drop=True)
pd.concat([y_orig, orig], axis=1).to_csv('original.csv', index=False)