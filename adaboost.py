import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


class AdaBoost:
    """
    An AdaBoost classifier.
    """

    def __init__(self, n_estimator=50, learning_rate=1):
        self.n_estimator = n_estimator
        self.learning_rate = learning_rate
        self.models = list()

    def fit(self, x, y):
        """
        Build a boosted classifier from the training set (x, y).

        parameters:
        -----------
        x : Features with the shape of (M, N) which M is the number of samples and N is number of features.
        y : Corresponding labels with the shape of (M,) which M is the number of samples. 
        """
        # Initial weight of samples
        m = x.shape[0]
        D = np.ones(m)/m

        t = 0
        while t < self.n_estimator:
            # Fit a classifire with specific weight
            h = tree.DecisionTreeClassifier(max_depth=1)
            h.fit(x, y, sample_weight=D)

            y_pred = h.predict(x)

            reverse_pred_mask = (y != y_pred).astype(int)

            e = np.sum(reverse_pred_mask * D)

            # Bad classifire
            if e > 0.5:
                D = np.ones(m)/m
                t -= 1
                continue

            a = 0.5 * np.log((1-e)/e)

            Z = 2 * ((e*(1-e))**(0.5))

            D = D * (np.exp(-a*y*y_pred) / Z)

            self.models.append((h, a))

            t += 1

    def predict(self, x):
        """
        Predict classes for x.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.
        """
        y_pred = np.zeros(x.shape[0])
        for model, a in self.models:
            y_pred += a*model.predict(x)
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        return y_pred

    def decision_function(self, x):
        """
        Compute the decision function of x.
        """
        y_pred = np.zeros(x.shape[0])
        for model, a in self.models:
            y_pred += a*model.predict(x)
        return y_pred

    def score(self, x, y):
        """
        computer score of classifire from the test set (x,y)
        """
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
