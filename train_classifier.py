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

from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)

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
        X_trn_s, y_trn_s, X_val_s, y_val_s = iterative_train_test_split(X, y, test_size=1-lab_prop)

        # Train RF classifier
        rf = RandomForestClassifier(n_estimators=1000,
                                    max_depth=15,
                                    random_state=1)
        rf.fit(X_trn, y_trn)
        y_hat = rf.predict(X_val)
        print("Random Forest, epoch:", config.feats_model[13:15], "labels prop:", lab_prop)
        print("Classification report: \n", (classification_report(y_val, y_hat)))
        print("ROC: ", (roc_auc_score(y_val, y_hat)))

        # Train RF classifier
        svm = OneVsRestClassifier(SVC(kernel='rbf',gamma='scale'))
        svm.fit(X_trn, y_trn)
        y_hat = svm.predict(X_val)
        print("SVM, epoch:", config.feats_model[13:15], "labels prop:", lab_prop)
        print("Classification report: \n", (classification_report(y_val, y_hat)))
        print("ROC: ", (roc_auc_score(y_val, y_hat)))

        # Train RF classifier
        rf = RandomForestClassifier(n_estimators=1000,
                                    max_depth=15,
                                    random_state=1)
        rf.fit(X_trn_s, y_trn_s)
        y_hat = rf.predict(X_val_s)
        print("Strat: Random Forest, epoch:", config.feats_model[13:15], "labels prop:", lab_prop)
        print("Classification report: \n", (classification_report(y_val_s, y_hat)))
        print("ROC: ", (roc_auc_score(y_val_s, y_hat)))

        # Train RF classifier
        svm = OneVsRestClassifier(SVC(kernel='rbf',gamma='scale'))
        svm.fit(X_trn_s, y_trn_s)
        y_hat = svm.predict(X_val_s)
        print("Strat: SVM, epoch:", config.feats_model[13:15], "labels prop:", lab_prop)
        print("Classification report: \n", (classification_report(y_val_s, y_hat)))
        print("ROC: ", (roc_auc_score(y_val_s, y_hat)))



