### Set Up ###

# global imports
from mlearn.synthetic import SyntheticRegression

### Main Program ###

# generate synthetic data
make_regression_kwargs = {'n_samples': 1000, 'n_features': 12}
continuous_range = [(-100, 100), (10, 20), (0, 50), (-5, 5), 
                    (-5, 60), (100, 2000), (4, 90), (10, 12), (0, 70)
                    ]
categories_number = [3, 4, 7]

synthetic_regression = SyntheticRegression(make_regression_kwargs=make_regression_kwargs,
                                           continuous_range=continuous_range, 
                                           categories_number=categories_number
                                          )