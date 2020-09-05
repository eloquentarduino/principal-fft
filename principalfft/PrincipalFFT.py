import numpy as np
from numpy.fft import rfft


class PrincipalFFT:
    """Extract principal FFT components for features generation"""
    def __init__(self, n_components):
        self.n_components = n_components
        self.idx = None

    def fit(self, X):
        """Fit data (find the n top components of the FFT)"""
        Xfft = rfft(X)
        Xabs = np.abs(Xfft)
        Xmax = Xabs.max(axis=0)
        self.idx = (-Xmax).argsort()[:self.n_components]
        return self

    def transform(self, X):
        """Transform data"""
        assert self.idx is not None, "PrincipalFFT instance not fitted"
        return np.abs(rfft(X)[:, self.idx])

    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)