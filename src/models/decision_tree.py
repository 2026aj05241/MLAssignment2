from sklearn.tree import DecisionTreeClassifier
from src.config import SEED
from src.preprocessing import make_pipeline

NAME = 'Decision Tree'

def build():
    return make_pipeline(DecisionTreeClassifier(max_depth=12, min_samples_leaf=3,
                                               random_state=SEED), scale=False)
