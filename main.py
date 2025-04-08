'''
pip install nltk
pip install scikit-learn
pip install numpy
pip install pandas
pip install wordcloud
pip install gensim
'''

import os
os.makedirs('Data Visualization', exist_ok=True)
os.makedirs('Results', exist_ok=True)
os.makedirs('Saved Data', exist_ok=True)


from datagen import datagen
from save_load import *
from Detection import MLRK, SVM, RF, DTree, NaiveBayes, KNN, PROPOSED_Hard
from plot_result import plotres
import matplotlib.pyplot as plt
import numpy as np
from I_SBOA import I_SBOA, fit_func


def full_analysis():
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70), (x_train_80, y_train_80, x_test_80, y_test_80)]

    i = 70

    for train_data in training_data:
        x_train, y_train, x_test, y_test = train_data

        # feature selection

        lb = np.zeros(x_train.shape[1])
        ub = np.ones(x_train.shape[1])

        pop_size = 5
        prob_size = len(lb)

        epochs = 100

        best_solution, best_fitness = I_SBOA(fit_func, lb, ub, pop_size, prob_size, epochs, i)

        soln = np.round(best_solution)
        selected_indices = np.where(soln == 1)[0]
        if len(selected_indices) == 0:
            selected_indices = np.where(soln == 0)[0]
            if len(selected_indices) == 0:
                selected_indices = np.random.randint(0, len(x_test), len(x_test) - 10)

        X_train = x_train[:, selected_indices]
        X_test = x_test[:, selected_indices]

        pred, met = MLRK(X_train, y_train, X_test, y_test)  # Soft Voting
        save(f'proposed_{i}', met)
        save(f'predicted_{i}', pred)

        pred, met = PROPOSED_Hard(X_train, y_train, X_test, y_test)  # Hard Voting
        save(f'proposed_hard_{i}', met)

        pred, met = SVM(X_train, y_train, X_test, y_test)
        save(f'svm_{i}', met)

        pred, met = NaiveBayes(X_train, y_train, X_test, y_test)
        save(f'naive_Bayes_{i}', met)

        pred, met = RF(X_train, y_train, X_test, y_test)
        save(f'rf_{i}', met)

        pred, met = DTree(X_train, y_train, X_test, y_test)
        save(f'dtree_{i}', met)

        pred, met = KNN(X_train, y_train, X_test, y_test)
        save(f'knn_{i}', met)


a = 0
if a == 1:
    full_analysis()

plotres()
plt.show()


