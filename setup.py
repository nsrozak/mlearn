# global imports
from setuptools import setup, find_packages

# library specifications
setup(name='mlearn',
      version='0.0.0',
      packages=find_packages(exclude=['examples', 'mlearn_tests']),
      install_requires=['scikit-learn==1.3.0',
                        'pandas==2.0.3'
                        ]
      )