from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


X, y = load_digits(return_X_y = True) # tridy??

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(max_iter = 1000, hidden_layer_sizes = (100, 80), alpha = 0.001, solver = "adam", random_state = 2)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

plt.matshow(X[0].reshape(8, 8), cmap = plt.cm.gray)
plt.xticks(()) # remove x tick marks
plt.yticks(()) # remove y tick marks
plt.show()

j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap = plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

x = X_test[5]

print(X.shape, y.shape)
print(X[0].reshape(8, 8))

print(mlp.predict([x]))
print(f"I predict that the number is... {mlp.predict([x])[0]}.")
print("accuracy:", mlp.score(X_test, y_test))

