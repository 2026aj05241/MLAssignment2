from sklearn.svm import SVC
from src.config import SEED
from src.preprocessing import make_pipeline

NAME = 'Support Vector Machine'

def build():
    return make_pipeline(SVC(C=3.0, kernel='rbf', gamma='scale', probability=True,
                             random_state=SEED, cache_size=256))
