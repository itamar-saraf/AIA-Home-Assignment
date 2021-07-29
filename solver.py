import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from dataset import Dataset
import numpy as np


class Solver:
    """
    Solver is trying to find the best classifier from a list of classifiers.
    Using Cross-validation
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.classifiers = [LogisticRegression, SVC, MultinomialNB]
        self.best_classifier = None

    def find_best_classifier(self):
        """
        Iterate over all classifiers and trying to find the best classifier after 5 runs.
        Using the mean score
        """
        all_scores = []
        for idx, classifier in enumerate(self.classifiers):
            clf = classifier()
            scores = cross_val_score(clf, np.matrix(self.dataset.train['ngrams'].tolist()),
                                     self.dataset.train['label_code'], cv=5)
            all_scores.append(scores.mean())
        best_clf_idx = np.argmax(all_scores)
        print(f'The Best Classifier is {self.classifiers[best_clf_idx].__name__}.'
              f'\nWith the mean score of {all_scores[best_clf_idx]}')
        self.best_classifier = self.classifiers[best_clf_idx]()

    def train_classifier(self):
        """
        Training the best classifiers that has been found
        """
        self.best_classifier.fit(np.matrix(self.dataset.train['ngrams'].tolist()),
                                 self.dataset.train['label_code'])

    def eval_on_test(self):
        """
        Evaluate the test set with the best classifier.
        Print statistics on the pred
        """
        x = self.dataset.test['tweet']
        x = self.dataset.cv.transform(x).toarray()
        y_pred = self.best_classifier.predict(x)
        self._print_stats(y_pred)

    def _print_stats(self, y_pred):
        """
        Prints F1, recall, precision, accuracy and confusion_matrix on the test set.
        :param y_pred: prediction on the dataset
        """
        cm = confusion_matrix(self.dataset.test['label_code'], y_pred)
        print(classification_report(self.dataset.test['label_code'], y_pred, target_names=self.dataset.le.classes_))

        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True)
        plt.show()

    def save_model(self):
        """
        Saving the best model with the count vectorizer(contains our vocab)
        to transforms more tweet in the future to arrays.
        Saves also the label encoding to reverse transforms he code labeling to language labelling
        """
        filename = 'finalized_model.sav'
        with open(filename, 'wb') as fout:
            pickle.dump((self.dataset.cv, self.dataset.le, self.best_classifier), fout)
