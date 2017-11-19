# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:45:56 2015

@author: Henry
"""

import numpy as np

class Classifier(object):

    def __init__(self, x_train, y_train, x_test, y_true):
        """Stores the classifier inputs.
        """

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_true = y_true

        self.y_pred = []


    def fit(self):
        raise NotImplementedError()

    def pred(self, x_test):
        raise NotImplementedError()

    def test(self, x_test=None, y_true=None):
        """Returns metrics based on the results of prediction (y_pred) in
        comparison to the y_true vector.

        If x_test or y_true is not defined, the test will consider
        the inputs given in the __init__ method.
        """
        y_pred = None

        #if a new case of test
        if x_test and y_true:
            y_pred = self.pred(x_test)
        else:
            y_pred = self.pred(self.x_test)
            y_true = self.y_true

        return y_pred


# You get this warning because you are using the f1-score, recall and precision without defining how they should be computed! The question could be rephrased: from the above classification report, how do you output one global number for the f1-score? You could:

# Take the average of the f1-score for each class: that's the avg / total result above. It's also called macro averaging.
# Compute the f1-score using the global count of true positives / false negatives, etc. (you sum the number of true positives / false negatives for each class). Aka micro averaging.
# Compute a weighted average of the f1-score. Using 'weighted' in scikit-learn will weigh the f1-score by the support of the class: the more elements a class has, the more important the f1-score for this class in the computation.
# These are 3 of the options in scikit-learn, the warning is there to say you have to pick one. So you have to specify an average argument for the score method.

# Which one you choose is up to how you want to measure the performance of the classifier: for instance macro-averaging does not take class imbalance into account and the f1-score of class 1 will be just as important as the f1-score of class 5. If you use weighted averaging however you'll get more importance for the class 5.

# The whole argument specification in these metrics is not super-clear in scikit-learn right now, it will get better in version 0.18 according to the docs. They are removing some non-obvious standard behavior and they are issuing warnings so that developers notice it.





#================================= SVM - RBF =================================
from sklearn import svm
class SVM_rbf(Classifier):
    name = "SVM_RBF"
    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = svm.SVC(kernel='rbf', )
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "SVM_RBF"


#=============================== SVM - Linear =================================
class SVM_linear(Classifier):
    name = "SVM_Linear"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = svm.LinearSVC()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "SVM_Linear"
#=========================== SVM - Polynomial =================================
class SVM_poly(Classifier):
    name = "SVM_Poly"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = svm.SVC(kernel='poly')
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "SVM_Poly"


#======================  Gaussian Naive Bayes =================================

from sklearn.naive_bayes import GaussianNB
class GaussianNaiveBayes(Classifier):
    name = "Gaussian Naive Bayes"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = GaussianNB()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)


    def __str__(self):
        return "Gaussian Naive Bayes"

#======================= Multinomial Naive Bayes ==============================
from sklearn.naive_bayes import MultinomialNB
class MultinomialNaiveBayes(Classifier):
    name = "Multinomial Naive Bayes"
    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = MultinomialNB()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "Multinomial Naive Bayes"


#============================= PLS regression =================================
from sklearn.cross_decomposition import PLSRegression
class PLS(Classifier):
    name = "PLS"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = PLSRegression()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "PLS"
#======================================= NSC =================================
from sklearn.neighbors.nearest_centroid import NearestCentroid
class NSC(Classifier):
    name =  "NSC"
    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = NearestCentroid()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "NSC"


#============================== Decision Tree =================================
from sklearn import tree

class DecisionTree(Classifier):
    name = "Decision Tree"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "Decision Tree"



#============================== Random Forest =================================
from sklearn.ensemble import RandomForestClassifier

class RandomForest(Classifier):
    name = "Random Forest"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = RandomForestClassifier()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "Random Forest"

#======================Linear Discriminant Analysis============================
# from sklearn.lda import LDA
# class LinearDA(Classifier):
#     name = "LDA"

#     def fit(self):
#         """Trains the model using the inputs from __init__ method.
#         """
#         self.model = LDA()
#         self.model.fit(self.x_train, self.y_train)

#     def pred(self, x_test):
#         """Predicts the classes of given samples.
#         """
#         return self.model.predict(x_test)

#     def __str__(self):
#         return "LDA"


#============================== AdaBoost =================================
from sklearn.ensemble import AdaBoostClassifier
class AdaBoost(Classifier):
    name = "AdaBoost"

    def fit(self):
        """Trains the model using the inputs from __init__ method.
        """
        self.model = AdaBoostClassifier()
        self.model.fit(self.x_train, self.y_train)

    def pred(self, x_test):
        """Predicts the classes of given samples.
        """
        return self.model.predict(x_test)

    def __str__(self):
        return "AdaBoost"






























