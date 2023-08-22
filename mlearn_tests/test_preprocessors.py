### Set Up ###

# global imports
import unittest
import pandas as pd
from mlearn.preprocessors import MLTrainPreprocessor

### Classes ###

class TestMLTrainPreprocessor(unittest.TestCase):
    def setUp(self):
        # create argumements
        self.data = pd.DataFrame([['A', 'X', 244, 5, 0.49, 9], 
                                  ['A', 'X', 148, 2, 0.32, 1], 
                                  ['A', 'Y', 470, 15, 0.46, 0], 
                                  ['B', 'Y', 11, 12, 0.55, 3], 
                                  ['B', 'Y', 320, 7, 0.1, 3], 
                                  ['C', 'Y', 51, 13, 0.02, 6], 
                                  ['C', 'Y', 289, 8, 0.21, 8], 
                                  ['C', 'Z', 398, 12, 0.68, 2], 
                                  ['D', 'Z', 77, 19, 0.18, 1], 
                                  ['E', 'Z', 178, 1, 0.52, 7]
                                  ], 
                                  columns=['col1', 'col2', 'col3', 'col4', 'col5', 'y']
                                 )
        self.y_column = 'y' 
        self.test_size = 0.3
        self.ohe_columns = ['col1', 'col2']
        self.ohe_kwargs = {'drop': 'first', 'sparse_output': False}
        self.scaler_columns = ['col3', 'col4', 'col5']
        self.scaler_kwargs = {}
        self.scaler_type = 'robust'
        self.random_state = 123

        # create instance of the class
        self.ml_train_preprocessor = MLTrainPreprocessor(self.data, self.y_column, 
                                                         test_size=self.test_size, 
                                                         ohe_columns=self.ohe_columns, 
                                                         ohe_kwargs=self.ohe_kwargs, 
                                                         scaler_columns=self.scaler_columns, 
                                                         scaler_kwargs=self.scaler_kwargs, 
                                                         scaler_type=self.scaler_type,
                                                         random_state=self.random_state
                                                         )

    def test_init(self): 
        # raise value error for incorrect scaler type
        scaler_type = 'scaler_type'

        with self.assertRaises(ValueError):
            MLTrainPreprocessor(self.data, self.y_column, 
                                test_size=self.test_size,
                                ohe_columns=self.ohe_columns, 
                                ohe_kwargs=self.ohe_kwargs, 
                                scaler_columns=self.scaler_columns, 
                                scaler_kwargs=self.scaler_kwargs, 
                                scaler_type=scaler_type,
                                random_state=self.random_state
                                )

        # confirm member variables are set correctly
        self.assertEqual(self.ml_train_preprocessor.y_column, self.y_column)
        self.assertEqual(self.ml_train_preprocessor.ohe_columns, self.ohe_columns)
        self.assertEqual(self.ml_train_preprocessor.scaler_columns, self.scaler_columns)

        # confirm new variables are set correctly
        self.assertTrue(self.ml_train_preprocessor.split_data)

    def test_get_data(self):
        # get the data
        X, y, X_test, y_test = self.ml_train_preprocessor.get_data()

        # check X
        self.assertEqual(X.shape[0], 7)
        self.assertEqual(X.columns.tolist(), ['col3', 'col4', 'col5', 'col1_B', 'col1_C', 
                                              'col1_D', 'col1_E', 'col2_Y', 'col2_Z'
                                             ]
                        )

        # check y
        self.assertEqual(len(y), 7)

        # check X_test
        self.assertEqual(X_test.shape[0], 3)
        self.assertEqual(X_test.columns.tolist(), ['col3', 'col4', 'col5', 'col1_B', 'col1_C', 
                                                   'col1_D', 'col1_E', 'col2_Y', 'col2_Z'
                                                  ]
                        )

        # check y_test
        self.assertEqual(len(y_test), 3)


### Main Program ###

if __name__ == '__main__':
    unittest.main()
    