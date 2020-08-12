import pandas as pd
import matplotlib.pyplot as plt

def plot_nn_performance(preds_out, preds_in, indx, dates, terms, y_orig, lags, folder, est = 'unified'):

    # Ordering predictions
    df_out = pd.DataFrame(data = preds_out, index = indx[1]).sort_index()
    df_in = pd.DataFrame(data = preds_in, index = indx[0]).sort_index()

    # Rebuilding testing set
    y_test = pd.DataFrame(data = y_orig.iloc[indx[1], :], index=indx[1]).sort_index()

    names = ["Normalized NCPI 4 Months Ahead", "Normalized Unemployment 4 Months Ahead", "Normalized NCPI 8 Months Ahead", "Normalized Unemployment 8 Months Ahead"]
    lbls = ["NCPI", "Unemployment", "NCPI", "Unemployment"]

    if est == 'mid_term':
        names = names[2:4]
        lbls = lbls[2:4]
    elif est == 'short_term':
        names = names[0:2]
        lbls = lbls[0:2]

    for k in range(0, len(terms)):

        # First plot
        fig = plt.figure(figsize = (10, 6))
        ax1 = fig.add_subplot(111)
        ax1.scatter(df_in.index, df_in.iloc[:,k], c='b', s =3, label='Fitted')
        plt.plot(y_orig.index, y_orig.iloc[:,k], c='black', label = 'Actual')
        sub_dates = dates[(lags+terms[k]):(len(dates)+terms[k])]
        plt.xticks(y_orig.index[::52], sub_dates[::52], rotation = 45)
        plt.title(names[k], fontsize = 15)
        plt.xlabel("Date", fontsize = 12)
        plt.ylabel(lbls[k], fontsize = 12)
        plt.legend()
        plt.savefig(folder + '/nn_all_' + y_orig.columns[k] + '_' + est + '.png')

        # Second plot
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        ax1.scatter(df_out.index, df_out.iloc[:, k], c='r', s=6, label='Predictions')
        plt.plot(y_orig.index, y_orig.iloc[:, k], c='skyblue', label='Actual')
        ax1.scatter(y_test.index, y_test.iloc[:, k], c='b', s=6, label='Test Set')
        plt.xticks(y_orig.index[::52], sub_dates[::52], rotation=45)
        plt.title(names[k], fontsize=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(lbls[k], fontsize=12)
        plt.legend()
        plt.savefig(folder + '/nn_test_' + y_orig.columns[k] + '_' + est  + '.png')


def plot_interest(solution, original, data_orig, y_orig, lags, folder='figs'):

    plt.figure(figsize = (15, 7.5))
    dates = data_orig.dates[lags:(len(solution) + lags)]
    plt.xticks(y_orig.index[::52], dates[::52], rotation=45)
    plt.plot(solution, color = 'skyblue', label = 'Solution')
    plt.plot(original, color = 'red', label = 'Real')
    plt.title('SELIC Rate Target')
    plt.legend()
    plt.xlabel("Date")
    plt.savefig(folder + '/optimal_results_meta_selic.png')
    plt.ylabel('Target')

def plot_sol(predictions, official, data_orig, lags, folder ='figs'):
    i = 0
    names = ["Normalized NCPI 4 Months Ahead", "Normalized Unemployment 4 Months Ahead",
             "Normalized NCPI 8 Months Ahead", "Normalized Unemployment 8 Months Ahead"]
    lbls = ["NCPI", "Unemployment", "NCPI", "Unemployment"]

    if len(predictions) < len(official):
        official = official[:len(predictions)]

    for k in official.columns:
        plt.figure(figsize = (15, 7.5))
        plt.plot(predictions[k], color ='skyblue', label ='Predicted')
        plt.plot(official[k], color ='red', label ='Official')
        plt.title(names[i], fontsize = 15)
        plt.ylabel(lbls[i], fontsize = 12)
        i = i + 1
        dates = data_orig.dates[lags:len(official) + lags]
        inxs = official.index[0:len(official):52]
        plt.xticks(inxs, dates[::52], rotation=45)
        plt.xlabel("Date", fontsize = 12)
        plt.legend()
        plt.savefig(folder + '/results_' + str(k) + '.png')

def plot_evolution(optimal_values, folder = 'figs'):
    plt.figure(figsize = (15, 7.5))
    plt.plot(optimal_values, color = 'blue')
    plt.title('Genetic Algorithm Evolution', fontsize = 15)
    plt.xlabel('Generation', fontsize = 12)
    plt.ylabel('Value of O(.)', fontsize= 12)
    plt.savefig(folder + '/result_evolution' + '.png')