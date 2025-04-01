from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

clf = RandomForestClassifier(n_jobs=-1, random_state=38)


