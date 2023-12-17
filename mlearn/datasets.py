### Set Up ###

# global imports
import os
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
        num_rescale_continuous = len(continuous_range)
        self.continuous_features = ['feature_' + str(i) for i in range(num_rescale_continuous)]

        self.categories_number = categories_number
        num_categorical = len(categories_number)
        self.categorical_features = ['feature_' + str(i) for i in 
                                     range(num_rescale_continuous, 
                                           num_rescale_continuous + num_categorical
                                           )
                                    ]

    def _rescale_continuous(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Rescales continuous variables
        Args:
            X: original continuous data
        Returns:
            X: rescaled continuous data
        '''
        # rescale specified features
        for i, feature_range in enumerate(self.continuous_range):
            scaler = MinMaxScaler(feature_range=feature_range)
            scaler.fit(X[self.continuous_features[i]].to_numpy().reshape(-1, 1))
            X[self.continuous_features[i]] = scaler.transform(X[self.continuous_features[i]].to_numpy().reshape(-1, 1))

        # return rescaled data
        return X
    
    def _make_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Makes categorical variables
        Args:
            X: original non-categorical data
        Returns:
            X: binned categorical data
        '''
        # make categorical variables
        for i, n_bins in enumerate(self.categories_number):
            discritizer = KBinsDiscretizer(n_bins, encode='ordinal', strategy='kmeans', subsample=None)
            discritizer.fit(X[self.categorical_features[i]].to_numpy().reshape(-1, 1))
            X[self.categorical_features[i]] = discritizer.transform(X[self.categorical_features[i]].to_numpy().reshape(-1, 1))
            X[self.categorical_features[i]]  = [chr(ord('@') + int(x) + 1) for x in X[self.categorical_features[i]]]

        # return categorical data
        return X

    def enhance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Enhances features
        Args: 
            X: original data
        Returns:
            X: data with enhanced features
        '''
        # raise errors
        if (len(self.continuous_features) > 0) and (len(self.categorical_features) > 0):
            if X.shape[1] < len(self.continuous_features) + len(self.categorical_features):
                raise ValueError('`X` does not have enough columns for prespecified enhancements')
            
        # update features
        X = self._rescale_continuous(X)
        X = self._make_categorical(X)
            
        # return dataframe
        return X


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
        X = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(X.shape[1])])
        data = self.enhance_features(X)
        data['y'] = y
        self.data = data

    def get_data(self) -> pd.DataFrame:
        ''' Gets the data
        Returns:
            data: the dataframe of synthetic data
        '''
        return self.data
    
    def save_data(self, path: str) -> None:
        ''' Saves data as a csv
        Args:
            path: directory for saving objects
        '''
        self.data.to_csv(os.path.join(path, 'data.csv'), index=False)
    

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
        X = pd.DataFrame(X, columns=['feature_' + str(i) for i in range(X.shape[1])])
        data = self.enhance_features(X)
        data['y'] = y
        self.data = data

    def get_data(self) -> pd.DataFrame:
        ''' Gets the data
        Returns:
            data: the dataframe of synthetic data
        '''
        return self.data
    
    def save_data(self, path: str) -> None:
        ''' Saves data as a csv
        Args:
            path: directory for saving objects
        '''
        self.data.to_csv(os.path.join(path, 'data.csv'), index=False)
