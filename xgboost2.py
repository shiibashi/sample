import numpy
import pandas
import xgboost as xgb
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
import functools
import numpy
import json

def _trade_loss(y_true, y_pred, w=1):
    """
        true, pred, res
        0, 0, 0
        1, 0, 1
        0, 1, -1: more penalty
        0, 0, 0
    """
    residual = (y_true - y_pred).astype("float")
    grad = numpy.where(residual<0, -2.0*w*residual, -2.0*residual)
    hess = numpy.where(residual<0, 2.0*w, 2.0)
    return grad, hess

def trade_loss(w):
    def f(y_true, y_pred):
        return _trade_loss(y_true, y_pred, w=w)
    return f
	
class Trainer(object):

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.best_score = 999999999

    def _optimize(self, X_train, y_train, X_test, y_test, trial):
        n_estimators = trial.suggest_int('n_estimators', 0, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 50)
        gamma = trial.suggest_discrete_uniform("gamma", 0, 0.5, 0.1)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.001, 0.01, 0.001)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
        #colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

        xgboost_tuna = xgb.XGBClassifier(
            objective=trade_loss(20),
            n_estimators = n_estimators,
            max_depth = max_depth,
            gamma=gamma,
            min_child_weight = min_child_weight,
            learning_rate = learning_rate,
            scale_pos_weight = scale_pos_weight,
            subsample = subsample,
            #colsample_bytree = colsample_bytree,
        )
        xgboost_tuna.fit(X_train, y_train)
        y_pred = xgboost_tuna.predict(X_test)
        score = -trade_score(y_test, y_pred) # accやaucなど
        self.update(score, xgboost_tuna)
        return score

    def update(self, score, xgb):
        params = xgb.get_params()
        if self.best_score > score:
            prefix = str(-1 * score)[2:6]
            self.best_score = score
            self.best_params = params
            self.best_model = xgb
            self._write_json(self.best_params, "tmp/best_params_{}.json".format(prefix))
            xgb.save_model("tmp/best_model_{}.xgb".format(prefix))
            print("best model updated: score: {}, params: {}".format(score, params))
        else:
            pass

    def _write_json(self, d, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(d, f)

    def train(self, train_x, train_y, test_x, test_y, n_trials=20):
        #clf = xgb.XGBClassifier()
        study = optuna.create_study()
        study.optimize(functools.partial(self._optimize, train_x, train_y,
                                         test_x, test_y), n_trials=n_trials)
trainer = Trainer()
trainer.train(train_df[feature_columns], train_df[label_col],
            test_df[feature_columns], test_df[label_col], n_trials=200)
study
clf.save_model("best_model.xgb")

clf = xgb.XGBClassifier()
clf.load_model("data/xgboost_model/best_model_0159.xgb")
clf._le = xgb.compat.XGBLabelEncoder()
clf._le.fit(train_df[label_col])
train_df["pred_label"] = clf.predict(train_df[feature_columns])
test_df["pred_label"] = clf.predict(test_df[feature_columns])
validation_df["pred_label"] = clf.predict(validation_df[feature_columns])