# packages
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
# custom package
from utils.grid_baseline1 import run


# run baseline1
# linear model: logistic regression
# text rep: one-hot vect using tf-idf

# TODO: performing oversampling to balance datasets
# performing grid search to find best params

# predicting author gender (sex)
# will plot best params and scores below

# print("GENDER")
# run('gender')

print("AGE")

run('age')
