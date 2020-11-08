import numpy as np
from numpy.fft import rfft
from sklearn.feature_selection import SelectKBest, RFE


class KFFT:
    """
    Extract principal FFT components for features generation
    """
    def __init__(self, feature_selector):
        assert isinstance(feature_selector, SelectKBest) or isinstance(feature_selector, RFE), 'feature_selector MUST be one of SelectKBest or RFE'
        self.feature_selector = feature_selector
        self.original_size = 0
        self.idx = None

    def fit(self, X, y):
        """
        Fit data (find the k top components of the FFT)
        :param X: input data
        :param y: input target
        :return: self
        """
        Xfft = np.abs(rfft(X))
        self.feature_selector.fit(Xfft, y)
        self.original_size = len(X[0])
        self.idx = self.get_idx()
        return self

    def transform(self, X):
        """
        Transform data
        :param X: input data
        :return: transformed data
        """
        assert self.idx is not None, "KFFT instance not fitted"
        return np.abs(rfft(X)[:, self.idx])

    def fit_transform(self, X, y):
        """
        Shortcut for fit + transform
        :param X: input data
        :param y: input targets
        :return: transformed data
        """
        self.fit(X, y)
        return self.transform(X)

    def get_idx(self):
        """
        Get top k FFT components idx
        :return:
        """
        if isinstance(self.feature_selector, SelectKBest):
            idx = (-self.feature_selector.scores_).argsort()[:self.feature_selector.k]
            return np.sort(idx)
        if isinstance(self.feature_selector, RFE):
            idx = self.feature_selector.ranking_.argsort()[:self.feature_selector.n_features_to_select]
            return np.sort(idx)
        raise RuntimeError('Unknown feature extractor')