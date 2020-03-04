from sklearn.svm import SVC
import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

d = Data.load_my_dataset()

X, y = d.data, d.target

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=3,
                                                    stratify=y)

svm_model_linear = SVC(kernel='linear').fit(train_X, train_y)
svm_model_poly = SVC(kernel='poly').fit(train_X, train_y)
svm_model_rbf = SVC(kernel='rbf').fit(train_X, train_y)

svm_prediction_l = svm_model_linear.predict(test_X)
svm_prediction_p = svm_model_poly.predict(test_X)
svm_prediction_g = svm_model_rbf.predict(test_X)

# model accuracy for X_test
accuracy = svm_model_linear.score(test_X, test_y)
print("SVM Linear model accuracy:",accuracy*100)

accuracy = svm_model_poly.score(test_X, test_y)
print("SVM Poly model accuracy:",accuracy*100)

accuracy = svm_model_rbf.score(test_X, test_y)
print("SVM RBF model accuracy:",accuracy*100)


training_accuracy = []
test_accuracy = []

C_settings = range(1, 21)

for c in C_settings:
    # build the model
    clf = SVC(kernel = 'linear',C = c).fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))


print("Linear Train",training_accuracy)
print("Linear Test:",test_accuracy)

plt.plot(C_settings, training_accuracy, label="training accuracy")
plt.plot(C_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()
plt.show()


training_accuracy = []
test_accuracy = []

C_settings = range(1, 21)

for c in C_settings:
    # build the model
    clf = SVC(kernel = 'poly',C = c).fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))


print("Poly Train",training_accuracy)
print("Poly Test:",test_accuracy)

plt.plot(C_settings, training_accuracy, label="training accuracy")
plt.plot(C_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()
plt.show()


training_accuracy = []
test_accuracy = []

C_settings = range(1, 21)

for c in C_settings:
    # build the model
    clf = SVC(kernel = 'rbf',C = c).fit(train_X, train_y)
    # record training set accuracy
    training_accuracy.append(clf.score(train_X, train_y))
    # record generalization accuracy
    test_accuracy.append(clf.score(test_X, test_y))


print("Gussian Train",training_accuracy)
print("Gussian Test:",test_accuracy)

plt.plot(C_settings, training_accuracy, label="training accuracy")
plt.plot(C_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("C")
plt.legend()
plt.show()
