import Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

d = Data.load_my_dataset()

X, y = d.data, d.target

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=3,
                                                    stratify=y)
print("Labels for training and testing data")
print(train_y)
print(test_y)

print('All:', np.bincount(y) / float(len(y)) * 100.0)
print('Training:', np.bincount(train_y) / float(len(train_y)) * 100.0)
print('Test:', np.bincount(test_y) / float(len(test_y)) * 100.0)

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(train_X, train_y)

print("Test set predictions: {}".format(clf.predict(test_X)))

print("Test set accuracy: {:.2f}".format(clf.score(test_X, test_y)))

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("k_neighbors")
plt.legend()
plt.show()