import os
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_dir', default='', type=str, help='Path to bigearthnet triplets')
    parser.add_argument('--feats_model', default='', type=str, help='Path to bigearthnet triplets')
    config = parser.parse_args()

    # Get features
    X = np.load(os.path.join(config.feats_dir, f'X_{config.feats_model}.npy'))
    y = np.load(os.path.join(config.feats_dir, f'y_{config.feats_model}.npy'))

    # Split data
    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2)

    # Train RF classifier
    rf = RandomForestClassifier(n_estimators=1000,
                                max_depth=20,
                                random_state=1)
    rf.fit(X_trn, y_trn)
    y_hat = rf.predict_proba(X_val)
    print("AND EVAL", f1_score(y_val, y_hat), average='weighted')