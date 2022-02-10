# importing all important modules and functions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.metrics import  confusion_matrix as c_m

# loading of dataset using the load_digits function
X, y = load_digits(return_X_y=True)
# makes train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# users input for choosing of number of itterations and type of solver
iterace = int(input("Input integer bigger than 200: "))
solv = str(input("Choose adam, sgd or lbfgs: " ))

# train an MLPClassifier on the training set
mlp = MLPClassifier(random_state=2, max_iter = iterace, solver = solv)
mlp.fit(X_train, y_train)

# catches the warning and ignores Continuous Integration - minimum of iterrations is 200
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print('0-9 digits recognition score (MNIST):',mlp.score(X_test, y_test))

# predicts the number
y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

# prepares confusion matrix of predicted and actual labels
cm = c_m(y_test, y_pred)
actual = '  actual  '
predic = '            p r e d i c t e d '
labels = '      0  1  2  3  4  5  6  7  8  9'

# prints confusion matrix of results
print()
print(' '*11, 'confusion matrix')
print()
print(predic)
print(labels)

for row in range(10):
    print(actual[row], row, cm[row])
