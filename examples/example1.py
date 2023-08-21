### Set Up ###

# global imports
from mlearn.synthetic import SyntheticRegression
from mlearn.preprocessing import MLTrainPreprocessor

### Main Program ###

# generate synthetic data
make_regression_kwargs = {'n_samples': 1000, 
                          'n_features': 12,
                          'random_state': 123
                          }
continuous_range = [(-100, 100), (10, 20), (0, 50), (-5, 5), 
                    (-5, 60), (100, 2000), (4, 90), (10, 12), (0, 70)
                    ]
categories_number = [3, 4, 7]

synthetic_regression = SyntheticRegression(make_regression_kwargs=make_regression_kwargs,
                                           continuous_range=continuous_range, 
                                           categories_number=categories_number
                                          )
data = synthetic_regression.get_data()

# print current summary statistics
print('Summary statistics of original data')
print(data.describe())

# preprocess the synthetic data
y_column = 'y'
test_size = 0.3
ohe_columns = ['feature_9', 'feature_10', 'feature_11']
ohe_kwargs = {'drop': 'first', 'sparse_output': False}
scaler_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 
                  'feature_5', 'feature_6', 'feature_7', 'feature_8'
                 ]
scaler_kwargs = {}
scaler_type = 'robust'
random_state = 123

ml_training_preprocessor = MLTrainPreprocessor(data, y_column, 
                                               test_size=test_size, 
                                               ohe_columns=ohe_columns, 
                                               ohe_kwargs=ohe_kwargs, 
                                               scaler_columns=scaler_columns, 
                                               scaler_kwargs=scaler_kwargs, 
                                               scaler_type=scaler_type,
                                               random_state=random_state
                                              )

X, y, X_test, y_test = ml_training_preprocessor.get_data()
train = X.copy()
train['y'] = y
test = X_test.copy()
test['y'] = y_test

# print preprocessed summary statistics
print('Summary statistics of train data')
print(train.describe())

print('Summary statistics of test data')
print(test.describe())