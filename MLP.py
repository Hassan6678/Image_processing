from sklearn.neural_network import MLPClassifier
import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

d = Data.load_my_dataset()

X, y = d.data, d.target

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=3,
                                                    stratify=y)
mlp = MLPClassifier(random_state=0)
mlp.fit(train_X, train_y)

print("Accuracy on training set: {:.2f}".format(mlp.score(train_X, train_y)))
print("Accuracy on test set: {:.2f}".format(mlp.score(test_X, test_y)))

# compute the mean value per feature on the training set
mean_on_train = train_X.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = train_X.std(axis=0)

# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (train_X - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (test_X - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, train_y)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, train_y)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, test_y)))


mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, train_y)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, train_y)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, test_y)))

training_accuracy = []
test_accuracy = []

n_iter = [200,400,600,800,1000,1200,1400,1600,1800,2000]
ran = 20
for n in n_iter:
    # build the model
    clf = MLPClassifier(max_iter=n, random_state=ran)
    clf.fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))
    ran = ran + 20
print("Test Accuracy: ", test_accuracy)
print("Train Accuracy: ", training_accuracy)

plt.plot(n_iter, training_accuracy, label="training accuracy")
plt.plot(n_iter, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_iteration")
plt.legend()
plt.show()