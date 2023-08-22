### Set Up ###

# global imports
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from typing import Any, Tuple


### Classes ###

class MLModel():
    def __init__(self, model: Any, classification: bool=True):
        self.model = model
        self.classification = classification  

    def predict(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, float]:
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
              ) -> Tuple[pd.Series, float] :
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
        # confirm model is trained
        if self.trained == False:
            raise ValueError('`model` not yet trained')
        
        # get results
        return self.best_params_, self.best_score_, self.cv_results_
    
    def get_model(self) -> Any:
        return self.model
    