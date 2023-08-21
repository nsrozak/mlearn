### Set Up ###

### Classes ###

class MLModel():
    def __init__(self):
        to_do = True

class MLTrainModel(MLModel):
    def __init__(self):
        # initialize parent class
        super(MLTrainModel, self).__init__()

    def _grid_search():
        to_do = True


# have parameter distributions and all other ml args in the init
# then when given data for training, it does the grid search and trains the model
# then have another method for getting predictions and use the same method on the train data
# and use it on the test data as a separate call (all of the above in the same func)
# different class that deals with just the test data and the pretrained model

# also maybe include inference data or something that was generated separately?
# or have train and validation data gotten from the same dataset and test from another creation
# using the regression
# i like that idea