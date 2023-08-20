# global imports
from setuptools import setup

# library specifications
setup(name='mlearn',
      version='0.0.0',
      packages=['mlearn'],
      install_requires=['scikit-learn==1.3.0',
                        'pandas==2.0.3'
                        ]
      )