from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.metrics import  confusion_matrix as c_m

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

iterace = int(input("Zadej celé číslo větší než 200:"))
solv = str(input("Vyber zkratku: adam, sgd, lbfgs" ))

mlp = MLPClassifier(random_state=2, max_iter = iterace, solver = solv) #, hidden_layer_sizes = (vrstvy, uzly), alpha = 0.001, solver = solv, random_state = 2)
mlp.fit(X_train, y_train)
# this example won't converge because of resource usage constraints on
# our Continuous Integration infrastructure, so we catch the warning and
# ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)



print('0-9 digits recognition score (MNIST):',mlp.score(X_test, y_test))

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

cm = c_m(y_test, y_pred)
actual = '  actual  '
predic = '            p r e d i c t e d '
labels = '      0  1  2  3  4  5  6  7  8  9'

print()
print(' '*11, 'confusion matrix')
print()
print(predic)
print(labels)

for row in range(10):
    print(actual[row], row, cm[row])
