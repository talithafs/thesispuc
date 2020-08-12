import pickle
import time
from itertools import combinations
import operator
import pandas as pd
import numpy as np

# Auxiliary functions

def expand_array(array, mult):
    expanded = []
    for val in array:
        expanded += [val]*mult
    return expanded


def build_dataset(solution, estimators, X_orig, terms, lags=7, interval=7):

    spc = lags*2
    selic_indx = [8, 19 + spc, 30 + spc, 41 + spc, 52 + spc, 63 + spc, 74 + spc, 85 + spc]
    depnd_indx = range(11, 27)
    depnd_names = X_orig.columns[depnd_indx]
    selic_names = X_orig.columns[selic_indx]
    n = len(X_orig)

    solution_lags = expand_array(solution, interval)
    l = len(solution_lags)
    solution_lags = np.array(solution_lags).reshape(-1, 1)

    for k in range(1, len(selic_indx)):
        expanded_sol = expand_array(solution, interval)
        a = np.array(expanded_sol)[:(l-k)]
        X = np.array([X_orig[selic_names[k]][0:k]])
        temp_res = np.vstack([X.reshape(-1, 1), a.reshape(-1, 1)])
        solution_lags = np.hstack([solution_lags, temp_res])

    solution_lags = solution_lags[:n, :]

    X = X_orig.copy()
    for k in range(len(selic_indx)):
        X[selic_names[k]] = solution_lags[:, k]
        X[X_orig.columns[selic_indx[k] + 1]] = solution_lags[:, k]

    for row in range(len(X)):

        if len(estimators) == 1:
            future_vals = estimators[0].predict(X.loc[row, :].values.reshape(1, -1))
        else:
            future_vals_1 = estimators[0].predict(X.loc[row, :].values.reshape(1, -1))
            future_vals_2 = estimators[1].predict(X.loc[row, :].values.reshape(1, -1))
            future_vals = np.array(pd.concat([pd.DataFrame(future_vals_1), pd.DataFrame(future_vals_2)], axis=1))

        for lag in range(1, lags+1):
            try:
                X.loc[row + terms[0] + lag, depnd_names[2*(lag-1)]] = future_vals[0][0]
                X.loc[row + terms[1] + lag, depnd_names[2*(lag-1) + 1]] = future_vals[0][1]
                X.loc[row + terms[2] + lag, depnd_names[2*(lag-1)]] = future_vals[0][2]
                X.loc[row + terms[3] + lag, depnd_names[2*(lag-1) + 1]] = future_vals[0][3]
            except Exception as ex:
                print(ex)
                continue
    X = X.iloc[:len(X_orig), :]
    return X

def objective(result, weights, r):
    obj_val = 0
    for i in range(len(result)):
        obj_val += ((1 + r)**i)*((result.iloc[i, 0]**weights[0]) *
                                 (result.iloc[i, 1]**weights[1]) *
                                 (result.iloc[i, 2]**weights[2]) *
                                 (result.iloc[i, 3]**weights[3]))
    return obj_val

## Genetic Algorithm functions

def pick_breeders(solutions, weights, d_rate, n_breeders, estimators, y_orig, X_orig, terms, lags=7, interval=7, rand=False):

    scores = {}

    for key in solutions.keys():
        X = build_dataset(solutions[key], estimators, X_orig, terms, lags, interval)

        if len(estimators) == 1:
            preds = pd.DataFrame(estimators[0].predict(X), columns=[y_orig.columns])
        else:
            preds1 = estimators[0].predict(X)
            preds2 = estimators[1].predict(X)
            preds = pd.concat([pd.DataFrame(preds1), pd.DataFrame(preds2)], axis=1)
            preds.columns = y_orig.columns

        scores[key] = objective(preds, weights, d_rate)

    breeders = sorted(scores.items(), key=operator.itemgetter(1))

    if rand:
        rand_inx = np.random.randint(n_breeders-1, len(breeders))
        rand_brd = breeders[rand_inx]
        breeders = breeders[:(n_breeders-1)]
        breeders.append(rand_brd)
    else:
        breeders = breeders[:n_breeders]

    breeders_ids = [k for k, v in breeders]
    return breeders_ids, breeders


def cross_over(breeders, solutions):

    cross_over_combinations = list(combinations(breeders, 2))
    pair = 1
    new_generation = {}
    n = len(solutions[1])
    size_split = int(np.round(n/2))
    orig_inds = solutions[1].index

    for i, j in cross_over_combinations:

        try:
            parent_1_indexes = np.random.choice(orig_inds, size_split)
        except:
            orig_inds = range(len(solutions[1]))
            parent_1_indexes = np.random.choice(orig_inds, size_split)

        child_1 = []
        child_2 = []

        for k in orig_inds:
            if k in parent_1_indexes:
                child_1 += [solutions[i][k]]
                child_2 += [solutions[j][k]]
            else:
                child_1 += [solutions[j][k]]
                child_2 += [solutions[i][k]]

        new_generation[pair] = child_1
        new_generation[pair + 1] = child_2
        pair += 2

    return new_generation

def mutation(solutions, perc, avail_values):

    n = len(solutions[1])
    m = int(np.round(n*perc))
    orig_inds = list(range(n))

    for key in solutions.keys():

        gen_mut = np.random.choice(orig_inds, m)

        for gen in gen_mut:
            new_gen = np.random.choice(avail_values, 1)[0]
            solutions[key][gen] += new_gen
            #solutions[key][gen] = solutions[key][gen] + np.random.randint(-1,2,1,)*np.random.rand(1, )

    return solutions

def run_optimization(X_orig, y_orig, estimators, weights, d_rate, n_breeders, terms, mut_perc, selic_options, lags=7, interval=7, runs=1000):

    # First generation
    solutions = {}
    selic = X_orig['meta.selic'][::interval]
    l_selic = len(selic)
    solutions[1] = selic + np.random.randint(-1, 2, l_selic,)*np.random.rand(l_selic, )
    solutions[2] = selic + np.random.randint(-1, 2, l_selic,)*np.random.rand(l_selic, )
    solutions[3] = selic + np.random.randint(-1, 2, l_selic,)*np.random.rand(l_selic, )
    solutions[4] = selic + np.random.randint(-1, 2, l_selic,)*np.random.rand(l_selic, )
    solutions[5] = selic
    solutions[6] = selic + np.random.randint(-1, 2, l_selic,)*np.random.rand(l_selic, )

    optimal_values = []
    best_generation = []
    #best_individual = []

    start = time.time()

    for generation in range(runs):
        breeders_ids, breeders = pick_breeders(solutions, weights, d_rate, n_breeders, estimators, y_orig, X_orig, terms, lags, interval, rand=True)

        # Save the best results
        if best_generation == []:
            best_generation = [solutions, breeders[0][1], generation]
        # If the current generation is better then the best generation update the best generation
        elif breeders[0][1] <= best_generation[1]:
            best_generation = [solutions, breeders[0][1], generation]

        if generation - best_generation[2] > 0:
            solutions = best_generation[0]
        else:
            new_generation = cross_over(breeders_ids, solutions)
            solutions = mutation(new_generation, mut_perc, selic_options)

        optimal_values += [best_generation[1]]
        #best_individual = best_generation[0][breeders[0][0]]

        if mut_perc >= .03:
            mut_perc = mut_perc ** 1.001

        print(str(generation) + ": " + str(breeders[0][1]))
        print(" ** Best Value: " + ": " + str(best_generation[1]))

    inx = pick_breeders(best_generation[0], weights, d_rate, n_breeders, estimators, y_orig, X_orig, terms, lags, interval, rand=True)[0][0]
    solutions = expand_array(best_generation[0][inx], interval)
    X = build_dataset(best_generation[0][inx], estimators, X_orig, terms, lags, interval)

    elapsed = (time.time() - start) / 3600
    print('Elapsed time : ', elapsed)

    return solutions, optimal_values, X

def get_mas(X, window, lags):

    spc = lags * 2
    selic_indx = [8, 9, 19 + spc, 19 + spc + 1, 30 + spc, 30 + spc + 1, 41 + spc, 41 + spc + 1, 52 + spc, 52 + spc + 1, 63 + spc, 63 + spc + 1, 74 + spc, 74 + spc + 1, 85 + spc, 85 + spc + 1]
    X_ma = X.copy()

    for indx in selic_indx:
       X_ma.iloc[:, indx] = X.iloc[:, indx].rolling(window=window*lags).mean()

    return X_ma[(window*lags):]

def save_outcomes(solutions, optimal_values, X, prfx =''):

    with open(prfx + 'solutions.pickle', 'wb') as file:
        pickle.dump(solutions, file)

    with open(prfx + 'optimal_values.pickle', 'wb') as file:
        pickle.dump(optimal_values, file)

    with open(prfx + 'X.pickle', 'wb') as file:
        pickle.dump(X, file)

def get_outcomes(prfx=''):

    with open(prfx + 'solutions.pickle', 'rb') as file:
        solutions = pickle.load(file)

    with open(prfx + 'optimal_values.pickle', 'rb') as file:
        optimal = pickle.load(file)

    with open(prfx + 'X.pickle', 'rb') as file:
        X = pickle.load(file)

    return solutions, optimal, X