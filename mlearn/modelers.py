### Set Up ###

# global imports
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from typing import Any, Tuple


### Classes ###

class MLModel():
    '''
    Args:
        model: model with `.fit(X, y)` and `.predict(X)` methods
        classification: whether the model is classificaiton or regression
    Returns:
        MLModel: an instance of the class
    '''
    def __init__(self, model: Any, classification: bool=True):
        self.model = model
        self.classification = classification  

    def predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, float]:
        ''' generates predictions with the model
        Args:
            X: features dataframe
            y: response data
        Returns:
            y_hat: predicted response data
            score: comparison metric of the response and predicted response data
        '''
        # get predictions
        y_hat = self.model.predict(X)

        # print score
        if self.classification == True:
            score = balanced_accuracy_score(y, y_hat)
            print('Balanced accuracy: ', score)
        else:
            score = mean_squared_error(y, y_hat, squared=False)
            print('Root mean squared error: ', score)

        # return predictions
        return y_hat, score


class MLTrainModel(MLModel):
    '''
    Args:
        model: model with `.fit(X, y)` and `.predict(X)` methods
        classification: whether the model is classificaiton or regression
        n_splits: number of data splits for cross validation
        n_iter: number of grid search iterations that occur
        scoring: method used to score during grid search
        random_state: reproducibility number used for spliting the data and
            sampling from distributions
    Returns:
        MLTrainModel: an instance of the class
    '''
    def __init__(self, model: Any, classification: bool=True, n_splits: int=5, n_iter: int=50,
                 scoring: str='balanced_accuracy', random_state: int=123):
        # initialize parent class
        super(MLTrainModel, self).__init__(model, classification=classification)

        # set member variables
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.trained = False

    def _randomized_search(self, X: pd.DataFrame, y: pd.Series, param_distributions: dict={}) -> None:
        ''' performs grid search to find the best parameters
        Args:
            X: features dataframe
            y: response data
            param_distributions: distributions for hyperparameters tried during grid search
        '''
        # get cv
        if self.classification == True:
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)
        else:
            cv = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        # create object
        randomized_search = RandomizedSearchCV(self.model,
                                               param_distributions,
                                               cv=cv,
                                               n_iter=self.n_iter,
                                               scoring=self.scoring,
                                               random_state=self.random_state,
                                               n_jobs=-2
                                               )
        # fit randomized search
        randomized_search.fit(X, y)

        # save variables
        self.best_params_ = randomized_search.best_params_
        self.best_score_ = randomized_search.best_score_
        self.cv_results_ = randomized_search.cv_results_

    def train(self, X: pd.DataFrame, y: pd.Series, param_distributions: dict={}
              ) -> Tuple[pd.Series, float]:
        ''' trains the model
        Args:
            X: features dataframe
            y: response data
            param_distributions: distributions for hyperparameters tried during grid search
        Returns:
            y_hat: predicted response data
            score: comparison metric of the response and predicted response data
        '''
        # confirmed model not yet trained
        if self.trained == True:
            raise ValueError('`model` already trained')
        
        # run randomized search
        self._randomized_search(X, y, param_distributions=param_distributions)

        # fit the model
        self.model.set_params(**self.best_params_)
        self.model.fit(X, y)
        
        # set model to trained
        self.trained = True

        # get predictions
        y_hat, score = self.predict(X, y)
        return y_hat, score
        
    def get_randomized_search_results(self) -> Tuple[dict, float, dict]:
        ''' gets results from grid search
        Returns:
            best_params_: parameters that yielded the best score
            best_score_: best score on the hold-out set during grid search
            cv_results_: all results from grid search
        '''
        # confirm model is trained
        if self.trained == False:
            raise ValueError('`model` not yet trained')
        
        # get results
        return self.best_params_, self.best_score_, self.cv_results_
    
    def get_model(self) -> Any:
        ''' gets the (possibly trained) model
        Returns:
            model: the model
        '''
        return self.model
    