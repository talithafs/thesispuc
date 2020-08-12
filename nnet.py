import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn.neural_network import MLPRegressor
import math
import itertools as itr
import warnings

def scale(data):
    cols = data.columns.copy()
    scaler = pp.StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    with open('scaler.pickle', 'wb') as file:
        pickle.dump(scaler, file)

    return pd.DataFrame(data, columns=cols)

def get_scaler():

    with open('scaler.pickle', 'rb') as file:
        scaler = pickle.load(file)

    return scaler

def split(data, lags, terms, t_size, random = True):
    # Storing info about independent/dependent variables
    X_cols = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    y_cols = [9, 13]
    X_cols_names = data.columns[X_cols].copy()
    y_cols_names = data.columns[y_cols].copy()

    # Spliting the dataset between features and outputs
    X_orig = data.loc[lags:, X_cols_names].copy()

    # Inserting lags for each series in X and y
    X_orig = X_orig.reset_index(drop=True)
    for k in range(1, (lags + 1)):
        new_cols = {}
        for col in y_cols_names:
            new_cols[col] = col + '_' + str(k) + '_lag'
        y_lag = data.loc[(lags - k): (len(data) - k - 1), y_cols_names].reset_index(drop=True)
        y_lag.rename(columns=new_cols, inplace=True)
        X_orig = pd.concat([X_orig, y_lag], axis=1)

    X_orig = X_orig.reset_index(drop=True)
    for k in range(1, (lags + 1)):
        new_cols = {}
        for col in X_cols_names:
            new_cols[col] = col + '_' + str(k) + '_lag'
        x_lag = data.loc[(lags - k): (len(data) - k - 1), X_cols_names].reset_index(drop=True)
        x_lag.rename(columns=new_cols, inplace=True)
        X_orig = pd.concat([X_orig, x_lag], axis=1)

    # Creating leads for each series in y
    y_orig = pd.DataFrame()
    for t in pd.unique(terms):
        if t != 0:
            new_cols = {}
            for col in y_cols_names:
                new_cols[col] = col + '_' + str(t) + '_lead'
            y_lead = data.loc[(lags + t):, y_cols_names].reset_index(drop=True)
            y_lead.rename(columns=new_cols, inplace=True)
            y_orig = pd.concat([y_orig, y_lead], axis=1)

    if len(terms) > 2:
        leads = terms[2]
    else:
        leads = terms[0]

    n = len(data) - lags - leads

    # Spliting into training and testing sets
    y_orig = y_orig.iloc[:n, ]
    X_orig = X_orig.iloc[:n, ]

    test_size = int(np.round(t_size*n))

    if random:
        indx = range(n)
        indx = ms.train_test_split(indx, test_size = test_size)

        X = np.array(X_orig)
        y = np.array(y_orig)

        X_train = X[indx[0], :]
        y_train = y[indx[0], :]
        X_test = X[indx[1],:]
        y_test = y[indx[1], :]
    else:
        indx = n - test_size

        X_train = np.array(X_orig.iloc[:indx, :])
        y_train = np.array(y_orig.iloc[:indx, :])
        X_test = np.array(X_orig.iloc[indx:, :])
        y_test = np.array(y_orig.iloc[indx:, :])

        indx = [range(0, indx), range(indx, n)]


    return X_train, y_train, X_test, y_test, X_orig, y_orig, indx

def combs(neurons, layers):

    nhidden = []

    for i in range(len(layers)):
        c = list(itr.combinations_with_replacement(neurons,layers[i]))
        for j in range(len(c)):
            nhidden.append(list(c[j]))

    return nhidden

def print_metrics(y_true, y_predicted, n_parameters):

    # Compute R^2 and adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1) / (y_true.shape[0] - n_parameters) * (1 - r2)

    ## Print usual metrics and  R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))

def calculate_metrics(y_true, y_predicted, n_parameters):

    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1) / (y_true.shape[0] - n_parameters) * (1 - r2)
    rmse = sklm.mean_squared_error(y_true, y_predicted)
    return r2, r2_adj, rmse

def regression(estimator, X_train, y_train, X_test, y_test):

    activation = [estimator.best_params_['activation']]
    solver = [estimator.best_params_['solver']]
    neurons = list(np.unique(estimator.best_params_['hidden_layer_sizes']))
    layers = [len(estimator.best_params_['hidden_layer_sizes'])]

    return regression(X_train, y_train, X_test, y_test, neurons, layers, activation, solver)


def regression(X_train, y_train, X_test, y_test, neurons, layers, activation=[], solver =[]):

    # Preparing KFold nested cross-validation
    inside = ms.KFold(n_splits=10, shuffle=True)
    outside = ms.KFold(n_splits=10, shuffle=True)

    param_dict = {}
    param_dict['hidden_layer_sizes'] = combs(neurons,layers)

    if len(activation) == 0:
        param_dict['activation'] = ['logistic', 'tanh', 'relu']
    else:
        param_dict['activation'] = activation

    if len(solver) == 0:
        param_dict['solver'] = ['lbfgs', 'sgd', 'adam']
    else:
        param_dict['solver'] = solver

    NN = MLPRegressor()
    estimator = GridSearchCV(NN, param_dict, n_jobs=2, cv = inside, scoring='neg_mean_squared_error')
    estimator.fit(X_train, y_train)

    cv_estimate = ms.cross_val_score(estimator, X_train, y_train, cv=outside)  # Use the outside folds
    score = -np.mean(cv_estimate)

    preds_out = estimator.predict(X_test)
    preds_in = estimator.predict(X_train)

    metrics = calculate_metrics(y_test, preds_out, X_test.shape[1])
    testing_metrics = {'R2':metrics[0], 'R2 Adj':metrics[1], 'RMSE':metrics[2]}

    metrics = calculate_metrics(y_train, preds_in, X_train.shape[1])
    training_metrics = {'R2': metrics[0], 'R2 Adj': metrics[1], 'RMSE': metrics[2], 'CV Mean RMSE': score}

    return preds_in, preds_out, estimator, testing_metrics, training_metrics

def save_estimators(estimators, sufx=''):

    if len(estimators) == 2:

        names = [str('st_estimator' + sufx + '.pickle'), str('mt_estimator' + sufx + '.pickle')]

        with open(names[0], 'wb') as file:
            pickle.dump(estimators[0], file)

        with open(names[1], 'wb') as file:
            pickle.dump(estimators[1], file)
    else:
        with open('un_estimator.pickle', 'wb') as file:
            pickle.dump(estimators[0], file)


def get_estimator(type='un', sufx=''):

    warnings.filterwarnings("ignore")

    name = str(type + '_estimator' + sufx + '.pickle')

    with open(name, 'rb') as file:
        estimator = pickle.load(file)

    return estimator



