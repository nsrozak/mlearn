### Set Up ###

# global imports
from mlearn.datasets import SyntheticRegression
from mlearn.preprocessors import MLTrainPreprocessor, MLPreprocessor

# training function arguments
train_make_regression_kwargs = {'n_samples': 1000, 
                                'n_features': 12,
                                'random_state': 123
                               }
continuous_range = [(-100, 100), (10, 20), (0, 50), (-5, 5), 
                    (-5, 60), (100, 2000), (4, 90), (10, 12), (0, 70)
                    ]
categories_number = [3, 4, 7]

y_column = 'y'
val_size = 0.3
ohe_columns = ['feature_9', 'feature_10', 'feature_11']
ohe_kwargs = {'drop': 'first', 'sparse_output': False}
scaler_columns = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 
                  'feature_5', 'feature_6', 'feature_7', 'feature_8'
                 ]
scaler_kwargs = {}
scaler_type = 'robust'
random_state = 123

# testing function arguments
test_make_regression_kwargs = {'n_samples': 100, 
                               'n_features': 12,
                               'random_state': 456
                              }

### Training ###

# generate synthetic data
train_synthetic_regression = SyntheticRegression(make_regression_kwargs=train_make_regression_kwargs,
                                                 continuous_range=continuous_range, 
                                                 categories_number=categories_number
                                                )
train_data = train_synthetic_regression.get_data()

# print current summary statistics
print('Summary statistics of original train data')
print(train_data.describe())

# preprocess the synthetic data
ml_training_preprocessor = MLTrainPreprocessor(train_data, y_column, 
                                               test_size=val_size, 
                                               ohe_columns=ohe_columns, 
                                               ohe_kwargs=ohe_kwargs, 
                                               scaler_columns=scaler_columns, 
                                               scaler_kwargs=scaler_kwargs, 
                                               scaler_type=scaler_type,
                                               random_state=random_state
                                              )

X, y, X_val, y_val = ml_training_preprocessor.get_data()
train = X.copy()
train['y'] = y
val = X_val.copy()
val['y'] = y_val

# print preprocessed summary statistics
print('Summary statistics of train data')
print(train.describe())

print('Summary statistics of validation data')
print(val.describe())

### Testing ###

# generate synthetic data
test_synthetic_regression = SyntheticRegression(make_regression_kwargs=test_make_regression_kwargs,
                                                continuous_range=continuous_range, 
                                                categories_number=categories_number
                                               )
test_data = test_synthetic_regression.get_data()

# print current summary statistics
print('Summary statistics of original test data')
print(test_data.describe())

# preprocess the synthetic data
ohe, ohe_columns, ohe_feature_names = ml_training_preprocessor.get_ohe()
scaler, scaler_columns = ml_training_preprocessor.get_scaler()

ml_preprocessor = MLPreprocessor(y_column, ohe=ohe, ohe_columns=ohe_columns, 
                                 ohe_feature_names=ohe_feature_names, 
                                 scaler=scaler, scaler_columns=scaler_columns
                                )

X_test, y_test = ml_preprocessor.preprocess_dataset(test_data)
test = X_test.copy()
test['y'] = y_test

# print preprocessed summary statistics
print('Summary statistics of test data')
print(test.describe())