

import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.optimize import fmin_l_bfgs_b
import warnings

from sklearn.neural_network.multilayer_perceptron import BaseEstimator, ClassifierMixin, RegressorMixin, \
    BaseMultilayerPerceptron
from sklearn.neural_network.multilayer_perceptron import is_classifier
from sklearn.neural_network.multilayer_perceptron import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.neural_network.multilayer_perceptron import SGDOptimizer, AdamOptimizer
from sklearn.neural_network.multilayer_perceptron import train_test_split
from sklearn.neural_network.multilayer_perceptron import LabelBinarizer
from sklearn.neural_network.multilayer_perceptron import gen_batches, check_random_state
from sklearn.neural_network.multilayer_perceptron import shuffle
from sklearn.neural_network.multilayer_perceptron import check_array, check_X_y, column_or_1d
from sklearn.neural_network.multilayer_perceptron import ConvergenceWarning
from sklearn.neural_network.multilayer_perceptron import safe_sparse_dot
from sklearn.neural_network.multilayer_perceptron import check_is_fitted
from sklearn.neural_network.multilayer_perceptron import _check_partial_fit_first_call, unique_labels
from sklearn.neural_network.multilayer_perceptron import type_of_target
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

_STOCHASTIC_SOLVERS = ['sgd', 'adam']


def _pack(coefs_, intercepts_):
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


class MLPClassifier_lucas(BaseMultilayerPerceptron, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10):

        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss='log_loss', shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)

    def _validate_input(self, X, y, incremental):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        if not incremental:
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        elif self.warm_start:
            classes = unique_labels(y)
            if set(classes) != set(self.classes_):
                raise ValueError("warm_start can only be used where `y` has "
                                 "the same classes as in the previous "
                                 "call to fit. Previously got %s, `y` has %s" %
                                 (self.classes_, classes))
        else:
            classes = unique_labels(y)
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(self.classes_)
            if len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError("`y` has classes not in `self.classes_`."
                                 " `self.classes_` has %s. 'y' has %s." %
                                 (self.classes_, classes))

        y = self._label_binarizer.transform(y)

        return X, y

    def predict(self, X):
        check_is_fitted(self, "coefs_")
        y_pred = self._predict(X)
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        return self._label_binarizer.inverse_transform(y_pred), y_pred

    def fit(self, X, y):
        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))

    @property
    def partial_fit(self):
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError("partial_fit is only available for stochastic"
                                 " optimizer. %s is not stochastic"
                                 % self.solver)
        return self._partial_fit

    def _partial_fit(self, X, y, classes=None):
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith('multilabel'):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        super()._partial_fit(X, y)

        return self

    def predict_log_proba(self, X):
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : array-like, shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, "coefs_")
        y_pred = self._predict(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred
