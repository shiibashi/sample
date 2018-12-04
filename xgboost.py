import pandas
import random
import numpy

from lib import util

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def train():
    train_data = _read()
    train_X, test_X, train_Y, test_Y = _split(train_data)
    print(train_X.columns)
    clf = xgb.XGBClassifier()
    clf_cv = GridSearchCV(clf,
         {'max_depth': [8,16,16,24], 'n_estimators': [100,100,200,200]}, verbose=1)
    clf_cv.fit(train_X, train_Y)

    clf = xgb.XGBClassifier(**clf_cv.best_params_)
    clf.fit(train_X, train_Y)
    pred = clf.predict_proba(test_X)
    pred_label = clf.predict(test_X)
    print(pred)
    print(confusion_matrix(test_Y, pred_label))
    print(classification_report(test_Y, pred_label))
    df = pandas.DataFrame(numpy.array(test_Y), columns=["label"])
    df = pandas.concat([df, pandas.DataFrame(pred, columns=["pred_0", "pred_1"])], axis=1)
    df["pred_label"] = pred_label
    df.to_csv("test.csv")
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tmp = df.copy()
        tmp["pred_label"] = df["pred_1"].apply(lambda x: 1 if x >= tau else 0)
        print("tau__{}".format(tau))
        #       pred0 pred1
        # true0
        # true1
        print(confusion_matrix(test_Y, tmp["pred_label"]))
        print("")

def _read():
    df = util.read_csv("preprocessor", "train_data.csv")
    df["profit_flag"] = df["profit"].apply(lambda x: 1 if x >= 0 else 0)
    df = df.sort_values("date").reset_index(drop=True).drop(["date", "code", "profit"], axis=1)
    return df

def _split(df):
    df = df.copy()
    X = df[df.columns[:-1]]
    Y = df[df.columns[-1]]
    n = int(len(df) * 0.7)
    train_X, test_X = X[0:n], X[n:]
    train_Y, test_Y = Y[0:n], Y[n:]
    train, test = df[0:n], df[n:]
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    #train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    return train_X, test_X, train_Y, test_Y