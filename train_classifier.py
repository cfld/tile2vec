import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_dir', default='', type=str, help='Path to bigearthnet triplets')
    parser.add_argument('--feats_model', default='', type=str, help='Path to bigearthnet triplets')
    config = parser.parse_args()

    # Get features
    X = np.load(os.path.join(config.feats_dir, f'X_{config.feats_model}.npy'))
    y = np.load(os.path.join(config.feats_dir, f'y_{config.feats_model}.npy'))

    for lab_prop in [.01, .05, .1, .2, .8]:
        # Split data
        X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=1-lab_prop)

        # Train RF classifier
        rf = RandomForestClassifier(n_estimators=1000,
                                    max_depth=15,
                                    random_state=1)
        rf.fit(X_trn, y_trn)
        y_hat = rf.predict(X_val)
        print("Random Forest, epoch:", config.feats_model[13:15], "labels prop:", lab_prop)
        print("Classification report: \n", (classification_report(y_val, y_hat)))
        print("ROC: ", (roc_auc_score(y_val, y_hat)))




