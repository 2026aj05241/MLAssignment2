from sklearn.linear_model import LogisticRegression
from src.config import SEED
from src.preprocessing import make_pipeline

NAME = 'Logistic Regression'

def build():
    return make_pipeline(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
