### Set Up ###

# global imports
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

### Classes ###

class SyntheticDataset():
    def __init__(self, continuous_range: list=[], categories_number: list=[]):
        # initialize member variables
        self.continuous_range = continuous_range
        self.num_rescale_continuous = len(continuous_range)
        self.categories_number = categories_number
        self.num_categorical = len(categories_number)

    def _rescale_continuous(self, X: np.array) -> np.array:
        # rescale specified features
        for i, feature_range in enumerate(self.continuous_range):
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(X[:, i])
            X[:, i] = scaler.transform(X[:, i])

        # return rescaled data
        return X
    
    def _make_categorical(self, X: np.array) -> np.array:
        # make categorical variables
        for i, n_bins in enumerate(self.categories_number):
            discritizer = KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans')
            discritizer.fit(X[:, i])
            X[:, i] = discritizer.transform(X[:, i])

        # return categorical data
        return X

    def _enhance_features(self, X: np.array) -> np.array:
        # raise errors
        if (self.num_rescale_continuous > 0) and (self.num_categorical > 0):
            if X.shape[1] < self.num_rescale_continuous + self.num_categorical:
                raise ValueError('`X` does not have enough columns for prespecified enhancements')
            
        # rescale continuous features
        _X = X[:, :self.num_rescale_continuous].copy()
        _X = self._rescale_continuous(_X)
        X[:, :self.num_rescale_continuous] = _X

        # create categorical variables
        _X = X[:, self.num_rescale_continuous:self.num_rescaled_continuous + self.num_categorical].copy()
        _X = self._make_categorical(_X)
        X[:, self.num_rescale_continuous:self.num_rescaled_continuous + self.num_categorical] = _X
            
        # return array
        return X
    
    def make_dataframe(self, X: np.array, y: np.array) -> pd.DataFrame:
        # enhance features
        X = self._enhance_features(X)
        # make dataframe
        data = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(X.shape[1])])
        data['y'] = y
        return data


class SyntheticClassification(SyntheticDataset):
    def __init__(self, make_classification_kwargs: dict={}):
        # initialize parent class
        super(SyntheticClassification, self).__init__()

        # get data
        X, y = make_classification(*make_classification_kwargs)
        data = self.make_dataframe(X, y)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data
    

class SyntheticRegression(SyntheticDataset):
    def __init__(self, make_regression_kwargs: dict={}):
        # initialize parent class
        super(SyntheticRegression, self).__init__()

        # get data
        X, y = make_regression(*make_regression_kwargs)
        data = self.make_dataframe(X, y)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        return self.data