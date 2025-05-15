#K-Nearest-Neighbor Model
#Naive-Bayes Model

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split




df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)


#Specify Dependent Variable Columns
independent_var = df.iloc[0:,0:22].values
dependent_var = df.iloc[0:,22:].values
indep_train, indep_test, dep_train, dep_test = train_test_split(independent_var,dependent_var,
                                                                test_size=0.2, random_state=8,
                                                                shuffle=True)
print(indep_train)


def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0 = time.time()
    if verbose == False:
        model.fit(X_train, y_train.ravel(), verbose=0)
    else:
        model.fit(X_train, y_train.ravel())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    time_taken = time.time() - t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Time taken = {}".format(time_taken))

    print(classification_report(y_test, y_pred, digits=5))
    plot_confusion_matrix(model, X_test, y_test, cmap='Blues', normalize='all')
    plot_roc_curve(model, X_test, y_test)
    plt.show()
    return model, accuracy, roc_auc, time_taken


params_kn = {'n_neighbors':5, 'algorithm': 'kd_tree', 'n_jobs':4}
model_kn = KNeighborsClassifier(**params_kn)
model_kn, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_kn, indep_train, dep_train, indep_test, dep_test)


params_nb = {}

model_nb = GaussianNB(**params_nb)
model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_model(model_nb, indep_train, dep_train, indep_test, dep_test)
