# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
dataset = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=42)

clf = GaussianNB()

param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}

gs = GridSearchCV(clf, param_grid, cv=5)

gs.fit(X_train, y_train)

best_params = gs.best_params_

clf = GaussianNB(var_smoothing=best_params['var_smoothing'])

cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

print("Cross-validation accuracy: {:.2f}% (+/- {:.2f}%)".format(cv_scores.mean()*100, cv_scores.std()*100))

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print("Test set accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("Sensitivity: {:.2f}%".format(specificity*100))
print("Specificity: {:.2f}%".format(specificity*100))
