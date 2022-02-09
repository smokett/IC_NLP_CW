import xgboost as xgb
from sklearn.metrics import f1_score
import numpy as np

from data_analysis import Preprocessor


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

if __name__=='__main__':
    pre = Preprocessor()
    X = pre.get_tfidf_vectors(df=df_train, train=True)
    X_test = pre.get_tfidf_vectors(df=df_test, train=False)

    d_train = xgb.DMatrix(X[X.columns.drop('label')].values, X['label'].values)
    d_valid = xgb.DMatrix(X_test[X_test.columns.drop('label')].values, X_test['label'].values)
    imbalance_weight = X['label'].value_counts(normalize=True)[0]/X['label'].value_counts(normalize=True)[1]
    print("Imbalance Weight: ",imbalance_weight)
    xgb_params = {'eta': 0.05, 
                  'max_depth': 12, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.75,
                  #'min_child_weight' : 1.5,
                  'scale_pos_weight': imbalance_weight,
                  'objective': 'binary:logistic', 
                  # 'eval_metric': 'auc', 
                  'disable_default_eval_metric': 1,
                  'seed': 23,
                  'lambda': 1.5,
                  'alpha': .6
                 }
                 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000, watchlist, feval=f1_eval, verbose_eval=10, early_stopping_rounds=50)