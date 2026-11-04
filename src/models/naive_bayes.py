from sklearn.naive_bayes import GaussianNB
from src.preprocessing import make_pipeline

NAME = 'Naive Bayes'

def build():
    return make_pipeline(GaussianNB(var_smoothing=1e-9))
