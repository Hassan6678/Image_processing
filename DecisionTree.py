from sklearn.tree import DecisionTreeClassifier
import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

d = Data.load_my_dataset()

X, y = d.data, d.target

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=3,
                                                    stratify=y)

tree = DecisionTreeClassifier()
tree.fit(train_X, train_y)
print("Accuracy on training set: {:.3f}".format(tree.score(train_X, train_y)))
print("Accuracy on test set: {:.3f}".format(tree.score(test_X, test_y)))