from sklearn.ensemble import RandomForestClassifier
import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

d = Data.load_my_dataset()

X, y = d.data, d.target

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=3,
                                                    stratify=y)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(train_X, train_y)

print("Random Forest Accuracy on training set: {:.3f}".format(forest.score(train_X, train_y)*100))
print("Random Forest Accuracy on test set: {:.3f}".format(forest.score(test_X, test_y)*100))

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
n_estimate = [20,40,60,80,100,120,140,160,180,200]

for n in n_estimate:
    # build the model
    clf = RandomForestClassifier(n_estimators=n,random_state=0)
    clf.fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))
print("Test Accuracy: ", test_accuracy)
print("Test Accuracy: ", training_accuracy)

plt.plot(n_estimate, training_accuracy, label="training accuracy")
plt.plot(n_estimate, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_estimate")
plt.legend()
plt.show()