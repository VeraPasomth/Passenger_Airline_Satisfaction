# Data Preprocessing
from random import random
from mlxtend.plotting import plot_learning_curves
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from numpy.random import seed
from numpy.random import randn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from scipy.stats import wilcoxon
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV file and Shorten Column Name
df = pd.read_csv("airline_passenger_satisfaction.csv", index_col=False)
df.rename(columns={"Departure and Arrival Time Convenience": "DATC"}, inplace=True)

# remove ID column
df.drop('ID', axis=1, inplace=True)

# replace empty space in column labels
df.columns = [c.replace(' ', '_') for c in df.columns]

columns_for_label_encoding = ["Class", "Customer_Type", "Gender", "Type_of_Travel", "Satisfaction"]
labelencoder = LabelEncoder()
for column in columns_for_label_encoding:
    df[column] = df[column].astype(str)
    df[column] = labelencoder.fit_transform(df[column])

# Arrange column order
dataFrame = pd.DataFrame(df, columns=['Gender', 'Customer_Type', 'Type_of_Travel', 'Class',
                                      'Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay',
                                      'DATC', 'Ease_of_Online_Booking', 'Check-in_Service', 'Online_Boarding',
                                      'Gate_Location', 'On-board_Service', 'Seat_Comfort', 'Leg_Room_Service',
                                      'Cleanliness', 'Food_and_Drink', 'In-flight_Service', 'In-flight_Wifi_Service',
                                      'In-flight_Entertainment', 'Baggage_Handling', 'Satisfaction'])

outlierDf = pd.DataFrame(df, columns=['Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay'])
print(df.head(10))

df = df[df['Arrival_Delay'] <= 500]
mean = round(dataFrame['Arrival_Delay'].mean())

# Fill-in Missing Values with Mean
dataFrame['Arrival_Delay'] = dataFrame['Arrival_Delay'].fillna(mean)
# Export to CSV File
dataFrame.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')

# Load Preprocessed Data
new_df = pd.read_csv("Preprocessed_Airline_Passenger_Satisfaction.csv", index_col=False)
QuantitativeDF = pd.DataFrame(df, columns=['Age', 'Flight_Distance', 'Arrival_Delay'])
new_df = new_df.drop('Unnamed: 0', axis=1)

# Update Preprocessed CSV File
new_df = new_df.drop('Departure_Delay', axis=1)
new_df.to_csv('Preprocessed_Airline_Passenger_Satisfaction.csv')

# Specify Dependent Variable Columns
independent_var = df.iloc[0:, 0:22].values
dependent_var = df.iloc[0:, 22:].values.ravel()

indep_train, indep_test, dep_train, dep_test = train_test_split(independent_var, dependent_var,
                                                                test_size=0.2, random_state=8,
                                                                shuffle=True)

indep_train, indep_val, dep_train, dep_val = train_test_split(indep_train, dep_train,
                                                              test_size=0.2, random_state=8,
                                                              shuffle=True)


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
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.pink, normalize='all')
    plot_roc_curve(model, X_test, y_test)
    plt.show()

    return model, accuracy, roc_auc, time_taken





params_rf = {'verbose':2}

rf_grid_params = {'max_depth': [8,16],
                'min_samples_leaf': [1,2],
                'n_estimators': [10],
                'random_state': [123],
                'verbose':[2]}

model_rf = RandomForestClassifier(**params_rf)
print("Before Grid Search")
model_rf, accuracy_kn, roc_auc_kn, tt_kn = run_model(model_rf, indep_train, dep_train, indep_test, dep_test)
t0 = time.time()

print("starting grid-search for Random Forest...")
grid = GridSearchCV(model_rf, rf_grid_params, cv=5, scoring='roc_auc', return_train_score=False, verbose=2)
best_rf=grid.fit(indep_train,dep_train)
best_rf_params = grid.best_estimator_
print("best params", best_rf_params)
y_pred = grid.predict(indep_test)
y_pred_train = grid.predict(indep_train)
accuracy = accuracy_score(dep_test, y_pred)
roc_auc = roc_auc_score(dep_test, y_pred)
time_taken = time.time() - t0




print("Test Accuracy Score Random Forest : {:.3f}".format(accuracy_score(dep_test, y_pred)))
print("Train Accuracy Score Random Forest: {:.3f}".format(accuracy_score(dep_train, y_pred_train)))
#Plot the learning curves
plot_learning_curves(indep_train, dep_train, indep_test, dep_test, grid,scoring='accuracy',
                     train_marker='o',test_marker='+')
plt.show()
#gridSearch results
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc))
print("Time taken = {}".format(time_taken))
print(classification_report(dep_test, y_pred, digits=5))
plot_confusion_matrix(best_rf, indep_test, dep_test, cmap="Blues", normalize='all')
plot_roc_curve(best_rf, indep_test, dep_test)
plt.show()

Y_preds_rf = best_rf.best_estimator_.predict(indep_test)
Y_preds_train_rf = best_rf.best_estimator_.predict(indep_train)

print("Training Accuracy = {}".format(grid.predict(indep_train)))
print("Validation Accuracy = {}".format(grid.predict(indep_val)))


def evaluate(params, grid,test_features, test_labels,model_name):
    predictions = params.predict(test_features)
    errors = abs(predictions - test_labels)
    accuracy = grid.best_score_
    print('Model Performance: ', model_name)
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

random_accuracy = evaluate(best_rf_params, grid,indep_test, dep_test,"Random Forest")




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


params_nb = {}

model_nb = CategoricalNB(**params_nb)
model_nb, accuracy_nb, roc_auc_nb, tt_nb = run_model(model_nb, indep_train, dep_train.ravel(), indep_test,
                                                     dep_test.ravel())

model_nb = CategoricalNB(min_categories=int(indep_train.max()))
model_nb.fit(indep_train, dep_train)

Y_preds = model_nb.predict(indep_test)

print(Y_preds[:15])
print(dep_test[:15])

print('\nTest Accuracy : {:.3f}'.format(model_nb.score(indep_test, dep_test)))
print('Training Accuracy : {:.3f}'.format(model_nb.score(indep_train, dep_train)))

print("Accuracy Score : {:.3f}".format(accuracy_score(dep_test, Y_preds)))
print("\nClassification Report :")
print(classification_report(dep_test, Y_preds))

n_features, n_classes, n_categories = independent_var.shape[1], np.unique(dependent_var), model_nb.n_categories_

params = {'alpha': [0.01, 0.1, 0.5, 1.0, ],
          'fit_prior': [True, False],
          'min_categories': [n_categories],
          'class_prior': [None, [0.1, ] * len(n_classes), ]
          }

print("starting Naive Bayes Grid-search")
best_nb = GridSearchCV(estimator=CategoricalNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5, error_score="raise")
best_nb.fit(indep_train, dep_train)

#Plot the learning curves
plot_learning_curves(indep_train, dep_train, indep_test, dep_test, best_nb,scoring='accuracy',
                     train_marker='o',test_marker='+')

print('Best Accuracy Through Grid Search : {:.3f}'.format(best_nb.best_score_))
print('Best Parameters : {}\n'.format(best_nb.best_params_))

Y_preds = best_nb.best_estimator_.predict(indep_test)
Y_preds_train = best_nb.best_estimator_.predict(indep_train)

print("Test Accuracy Score : {:.3f}".format(accuracy_score(dep_test, Y_preds)))
print("Train Accuracy Score : {:.3f}".format(accuracy_score(dep_train, Y_preds_train)))
print("\nClassification Report :")
print(classification_report(dep_test, Y_preds))


cm = confusion_matrix(dep_test, Y_preds)

cm = confusion_matrix(dep_test, Y_preds, labels=model_nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_nb.classes_)
disp.plot()
plt.show()

plot_roc_curve(best_nb, indep_test, dep_test)

val_preds = best_nb.best_estimator_.predict(indep_val)
best_vals = best_nb.best_estimator_
print("Validation Accuracy Score : {:.3f}".format(accuracy_score(dep_val, val_preds)))

cm = confusion_matrix(dep_val, val_preds)

nb_eval = evaluate(best_vals,best_nb,indep_test,dep_test,"Naive Bayes")
cm = confusion_matrix(dep_val, val_preds, labels=model_nb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_nb.classes_)
disp.plot()
plt.show()

plot_roc_curve(best_nb, indep_val, dep_val)


# Normalize Features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(indep_train)
X_test = scaler.fit_transform(indep_test)

# Hypothesis testing for statistical significance

print("__________________________Hypothesis Test_________________________________________")
seed(1)
scores1 = cross_val_score(best_nb, X_train, dep_train.ravel(), error_score='raise', scoring='accuracy', cv=10,
                          n_jobs=-1, verbose=2)

scores2 = cross_val_score(best_rf, X_train, dep_train.ravel(), scoring='accuracy', cv=10, n_jobs=-1, verbose=2)

print('NB cross_val training scores: ', scores1)
print('RF cross_val training scores: ', scores2)

# check if difference between algorithms performance
stat, p = wilcoxon(scores1, scores2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
#interpret
alpha = 0.05
if p > alpha:
    print('Performance is the same (fail to reject H0)')
else:
    print('There is enough evidence to conclude a difference in algorithm performance(reject H0)')



