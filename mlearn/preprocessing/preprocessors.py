### Set Up ###

# global imports
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from typing import Optional, Union, Tuple, Any

# create types
Scaler = Union[StandardScaler, RobustScaler, MinMaxScaler]

### Classes ###

class MLPreprocessor():
    '''
    Args:
        y_column: response column in the dataset
        ohe: one-hot encoder object
        ohe_columns: columns that are one-hot encoded
        scaler: scaler object
        scaler_columns: columns that are scaled
    Returns:
        MLPreprocessor: an instance of the class
    '''
    def __init__(self, y_column: str, ohe: Optional[OneHotEncoder]=None, ohe_columns: list=[], 
                 scaler: Optional[Scaler]=None,  scaler_columns: list=[]):
        # raise errors
        if y_column in ohe_columns:
            raise ValueError('`y_column` cannot be one-hot encoded')
        if y_column in scaler_columns:
            raise ValueError('`y_column` cannot be scaled')

        # member variables
        self.y_column = y_column

        # ohe member variables
        self.ohe = ohe
        self.ohe_columns = ohe_columns

        # scaler member vairiables
        self.scaler = scaler
        self.scaler_columns = scaler_columns

    def _separate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        ''' Separates data into X and y
        Args:
            data: dataset
        Returns
            X: features dataframe
            y: response data
        '''
        X = data.drop(columns=self.y_column)
        y = data[self.y_column]
        return X, y

    def _ohe_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' One hot encodes `ohe_columns`
        Args: 
            X: features dataframe
        Returns:
            X: features dataframe with one-hot encoded columns
        '''
        X_ohe_series = self.ohe.transform(X[self.ohe_columns])
        X_ohe = pd.DataFrame(X_ohe_series, index=X.index)
        X = pd.concat([X.drop(columns=self.ohe_columns), X_ohe], axis=1)
        return X
    
    def _scaler_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ''' Scales `scaler_columns`
        Args:
            X: features dataframe
        Returns:
            X: features dataframe with scaled columns
        '''
        X[self.scaler_columns] = self.scaler.transform(X[self.scaler_columns])
        return X
    
    def preprocess_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        ''' Preprocesses a dataset
        Args:
            data: dataset
        Returns
            X: features dataframe
            y: response data
        '''
        # split data into X and y
        X, y = self._separate_data(data)
        # one-hot encode data
        if self.ohe is not None:
            X = self._ohe_transform(X)
        # scale data
        if self.scaler is not None:
            X = self._scaler_transform(X)
        # return data
        return X, y


class MLTrainPreprocessor(MLPreprocessor):
    '''
    Args:
        data: dataset
        y_column: response column in the dataset
        test_size: amount of data used for testing
        ohe_columns: columns that are one-hot encoded
        ohe_kwargs: keyword arguments passed to one-hot encoder object
        ohe_feature_names: feature names for columns after ohe-hot encoding
        scaler_columns: columns that are scaled
        scaler_kwargs: keyword arguments passed to scaler object
        scaler_type: type of scaler, one of {'robust', 'standard', 'minmax'}
        random_state: reproducibility number used for splitting data
    Returns:
        MLTrainPreprocessor: an instance of the class
    '''
    def __init__(self, data: pd.DataFrame, y_column: str, test_size: Optional[float]=None, 
                 ohe_columns: list=[], ohe_kwargs: dict={'drop': 'first', 'sparse': False}, 
                 scaler_columns: list=[], scaler_kwargs: dict={}, scaler_type: str='robust',
                 random_state: Optional[int]=123):
        # raise errors
        if scaler_type not in ['robust', 'standard', 'minmax']:
            raise ValueError('`scaler_type` is not supported, must be one of `robust`, `standard`, `minmax`')

        # initialize parent class
        super(MLTrainPreprocessor, self).__init__(y_column=y_column,
                                                  ohe_columns=ohe_columns, 
                                                  scaler_columns=scaler_columns
                                                 )

        # determine if data is split
        self.split_data = True if test_size != None else False

        # split data into train and test
        if self.split_data == True:
            train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        else:
            train = data.copy()

        # fit quantities
        if len(ohe_columns) > 0:
            self._ohe_fit(train, ohe_kwargs=ohe_kwargs)

        if len(scaler_columns) > 0:
            self._scaler_fit(train, scaler_kwargs=scaler_kwargs)

        # preprocess data
        X, y = self.preprocess_dataset(train)

        if self.split_data == True:
            X_test, y_test = self.preprocess_dataset(test)

        # set member variables
        self.X = X
        self.y = y

        if self.split_data == True:
            self.X_test = X_test
            self.y_test = y_test

    def _ohe_fit(self, data: pd.DataFrame, ohe_kwargs: dict={}) -> None:
        ''' Fits a one-hot encoder for `ohe_columns`
        Args:
            data: dataset
            ohe_kwargs: keyword arguments passed to one-hot encoder object
        '''
        # fit the one hot encoder
        ohe = OneHotEncoder(**ohe_kwargs)
        ohe.fit(data[self.ohe_columns])
        self.ohe = ohe

    def _scaler_fit(self, data: pd.DataFrame, scaler_kwargs: dict={}, scaler_type: str='robust'
                    ) -> None:
        ''' Fits a scaler for `scaler_columns`
        Args:
            data: dataset
            scaler_kwargs: keyword arguments passed to scaler object
            scaler_type: type of scaler, one of {'robust', 'standard', 'minmax'}
        '''
        # create scaler
        if scaler_type == 'robust':
            scaler = RobustScaler(**scaler_kwargs)
        elif scaler_type == 'standard': 
            scaler = StandardScaler(**scaler_kwargs)
        else:
            scaler= MinMaxScaler(**scaler_kwargs)

        # fit scaler
        scaler.fit(data[self.scaler_columns])

    def get_ohe(self) -> Tuple[Optional[OneHotEncoder], Optional[list]]:
        ''' Gets one-hot encoder items
        Returns:
            ohe: one-hot encoder object
            ohe_columns: columns that are one-hot encoded
            ohe_feature_names: feature names for columns after ohe-hot encoding
        '''
        if self.ohe is not None:
            return self.ohe, self.ohe_columns
        else:
            return None, None
        
    def get_scaler(self) -> Tuple[Optional[Scaler], Optional[list]]:
        ''' Gets scaler items
        Returns:
            scaler: scaler object
            scaler_columns: columns that are scaled
        '''
        if self.scaler is not None:
            return self.scaler, self.scaler_columns
        else:
            return None, None

    def get_data(self) -> Tuple[Any, ...]:
        ''' gets data
        Returns
            X: train features dataframe
            y: train response data
            X_test: test features dataframe
            y_test: test response dataframe
        '''
        if self.split_data == True:
            return self.X, self.y, self.X_test, self.y_test
        else:
            return self.X, self.y
        