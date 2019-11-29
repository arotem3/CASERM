import numpy as np
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import accuracy_score, classification_report
import random
import statistics
from statistics import mode
from sklearn.model_selection import train_test_split

def predict_from_distribution(distribution):
    assert sum(distribution) == 1
    #labelz = list(range(len(distribution)))
    #return np.asarray(random.choices(labelz,weights = distribution))
    x = random.random()
    for i in range(len(distribution)):
      if x < distribution[0]:
        return 0
      elif x < sum(distribution[0: i+1]):
        return i

def select_most_common(labels):
    """Select the most common value in an iterable of labels

    Args:
        labels (iterable): An iterable of integers representing the labels of a dataset

    Returns:
        int: The most common element in the iterable
    """

    return mode(labels)


class NaiveClassifier:
    """A Naive Classifier that predicts classes using simple approaches.
    """

    def __init__(self, approach, value=None):
        """Initialize the NaiveClassifier

        Args:
            approach (str): One of "always", "most", "distribution", "equal"
            value (int, optional): Defaults to None. The value of the class to select if approach is "always"
        """
        assert approach in ["always", "most", "distribution", "equal"]
        self.approach = approach
        self.value = value

    def fit(self,X,y):
        """Fit to data and labels

        Args:
            X (iterable): The features of the data
            y (iterable): The labels of the data
        """
        if self.approach == "always":
            self.value = [value]
        elif self.approach == "most":
            # YOUR CODE HERE
            #mostcommon = []
            #mostcommon =  mostcommon.append(select_most_common(y))
            most_common = [select_most_common(y)]
            self.most_common = most_common
        elif self.approach == "distribution":
            # YOUR CODE HERE
            class0 = y.count(0)
            class1 = y.count(1)
            class2 = y.count(2)
            pclass0 = class0/(class0+class1+class2)
            pclass1 = class1/(class0+class1+class2)
            pclass2 = class2/(class0+class1+class2)
            distribution = [pclass0,pclass1,pclass2]
            self.distribution = distribution
        elif self.approach == "equal":
            # YOUR CODE HERE
            class0 = y.count(0)
            class1 = y.count(1)
            class2 = y.count(2)
            pclass0 = class0/(class0+class1+class2)
            pclass1 = class1/(class0+class1+class2)
            pclass2 = class2/(class0+class1+class2)
            l = np.array([pclass0,pclass1,pclass2])
            meanp = l[np.nonzero(l)].mean()
            distributioneq = [meanp,meanp,meanp]
            self.distributioneq = distributioneq

    def predict(self,X):
        """Predict the labels of a new set of datapoints

        Args:
            X (iterable): The data to predict
        """
        if self.approach == "always":
            value = self.value
            y_pred = [value]*len(X)
            return y_pred
        elif self.approach == "most":
            most_common = self.most_common
            y_pred = most_common*len(X)
            return y_pred
        elif self.approach == "distribution":
            distribution=self.distribution
            y_pred = [predict_from_distribution(distribution) for i in range(len(X))]
            return y_pred
        elif self.approach == "equal":
            distributioneq=self.distributioneq
            y_pred = [predict_from_distribution(distributioneq) for i in range(len(X))]
            return y_pred


def rand_array_tri(k, j, l, n):
    arg = np.zeros(n, dtype=int)
    arg[:k]  = 1
    arg[j:l] = 2
    np.random.shuffle(arg)
    return arg


y = rand_array_tri(2060,2136,5624, 5624)
X = np.random.randint(0,2,size = (5624,))

y = y.tolist()
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

always_zero = NaiveClassifier(approach="always", value=0)
always_one = NaiveClassifier(approach="always", value=1)
always_two = NaiveClassifier(approach="always", value=2)

most_est = NaiveClassifier(approach="most")
most_est.fit(X_train,y_train)
dist_est = NaiveClassifier(approach='distribution')
dist_est.fit(X_train,y_train)
equal_est = NaiveClassifier(approach='equal')
equal_est.fit(X_train,y_train)

pred_0 = always_zero.predict(X)
pred_1 = always_one.predict(X)
pred_2 = always_two.predict(X)

pred_most = most_est.predict(X)
pred_est = dist_est.predict(X)
pred_equal = equal_est.predict(X)

print('accuracy of predicting always the first class:', accuracy_score(y, pred_0))
print('accuracy of predicting always the second class:', accuracy_score(y, pred_1))
print('accuracy of predicting always the third class:' , accuracy_score(y, pred_2))

print('accuracy of predicting always the most common class:', accuracy_score(y, pred_most))
print('accuracy of predicting each class based on the distribution of the classes:', accuracy_score(y, pred_est))
print('accuracy of predicting each class equally:', accuracy_score(y, pred_equal))
