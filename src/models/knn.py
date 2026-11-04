from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing import make_pipeline

NAME = 'KNN'

def build():
    return make_pipeline(KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=1))
