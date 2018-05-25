from sklearn.linear_model import RidgeClassifier


class RidgeClassifierWithProba(RidgeClassifier):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001,
                 class_weight=None, solver='auto', random_state=None):
        super(RidgeClassifierWithProba, self).__init__(alpha, fit_intercept, normalize, copy_X, max_iter, tol,
                                                       class_weight, solver, random_state)

    def predict_proba(self, X):
        return self._predict_proba_lr(X)
