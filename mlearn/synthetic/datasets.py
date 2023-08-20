### Set Up ###

# global imports
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

### Classes ###

class SyntheticDataset():
    '''
    Args:
        continuous_range: range for each of the continuous variables, as [(low1, high1), ...]
        categories_number: number of categories for each categorical variable, as [num1, ...]
    Returns:
        SyntheticDataset: an instance of the class
    '''
    def __init__(self, continuous_range: list=[], categories_number: list=[]):
        # initialize member variables
        self.continuous_range = continuous_range
        self.num_rescale_continuous = len(continuous_range)
        self.categories_number = categories_number
        self.num_categorical = len(categories_number)

    def _rescale_continuous(self, X: np.array) -> np.array:
        ''' Rescales continuous variables
        Args:
            X: original continuous data
        Returns:
            X: rescaled continuous data
        '''
        # rescale specified features
        for i, feature_range in enumerate(self.continuous_range):
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(X[:, i].reshape(-1, 1))
            X[:, i] = np.squeeze(scaler.transform(X[:, i].reshape(-1, 1)))

        # return rescaled data
        return X
    
    def _make_categorical(self, X: np.array) -> np.array:
        ''' Makes categorical variables
        Args:
            X: original non-categorical data
        Returns:
            X: binned categorical data
        '''
        # make categorical variables
        for i, n_bins in enumerate(self.categories_number):
            discritizer = KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans', subsample=None)
            discritizer.fit(X[:, i].reshape(-1, 1))
            X[:, i] = np.squeeze(discritizer.transform(X[:, i].reshape(-1, 1)))

        # return categorical data
        return X

    def _enhance_features(self, X: np.array) -> np.array:
        ''' enhances features
        Args: 
            X: original data
        Returns:
            X: data with enhanced features
        '''
        # raise errors
        if (self.num_rescale_continuous > 0) and (self.num_categorical > 0):
            if X.shape[1] < self.num_rescale_continuous + self.num_categorical:
                raise ValueError('`X` does not have enough columns for prespecified enhancements')
            
        # rescale continuous features
        _X = X[:, :self.num_rescale_continuous].copy()
        _X = self._rescale_continuous(_X)
        X[:, :self.num_rescale_continuous] = _X

        # create categorical variables
        _X = X[:, self.num_rescale_continuous:self.num_rescale_continuous + self.num_categorical].copy()
        _X = self._make_categorical(_X)
        X[:, self.num_rescale_continuous:self.num_rescale_continuous + self.num_categorical] = _X
            
        # return array
        return X
    
    def make_dataframe(self, X: np.array, y: np.array) -> pd.DataFrame:
        ''' Makes a dataframe out of arrays
        Args:
            X: features data
            y: response data
        Returns:
            data: dataframe
        '''
        # enhance features
        X = self._enhance_features(X)
        # make dataframe
        data = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(X.shape[1])])
        data['y'] = y
        return data


class SyntheticClassification(SyntheticDataset):
    '''
    Args:
        make_classification_kwargs: keyword arguments for the make_classification function
        continuous_range: range for each of the continuous variables, as [(low1, high1), ...]
        categories_number: number of categories for each categorical variable, as [num1, ...]
    Returns:
        SyntheticClassification: an instance of the class
    '''
    def __init__(self, make_classification_kwargs: dict={}, continuous_range: list=[], 
                 categories_number: list=[]):
        # initialize parent class
        super(SyntheticClassification, self).__init__(continuous_range=continuous_range, 
                                                      categories_number=categories_number
                                                      )

        # get data
        X, y = make_classification(**make_classification_kwargs)
        data = self.make_dataframe(X, y)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        ''' Gets the data
        Returns:
            data: the dataframe of synthetic data
        '''
        return self.data
    

class SyntheticRegression(SyntheticDataset):
    '''
    Args:
        make_regression_kwargs: keyword arguments for the make_regression function
        continuous_range: range for each of the continuous variables, as [(low1, high1), ...]
        categories_number: number of categories for each categorical variable, as [num1, ...]
    Returns:
        SyntheticRegression: an instance of the class
    '''
    def __init__(self, make_regression_kwargs: dict={}, continuous_range: list=[], 
                 categories_number: list=[]):
        # initialize parent class
        super(SyntheticRegression, self).__init__(continuous_range=continuous_range, 
                                                  categories_number=categories_number
                                                 )

        # get data
        X, y = make_regression(**make_regression_kwargs)
        data = self.make_dataframe(X, y)
        self.data = data

    def get_data(self) -> pd.DataFrame:
        ''' Gets the data
        Returns:
            data: the dataframe of synthetic data
        '''
        return self.data