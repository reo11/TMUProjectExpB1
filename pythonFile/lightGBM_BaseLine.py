import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# iris = load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=0)
# print(X_train)
# print('\n')
# print(X_test)
# print('\n')
# print(y_train)
# print('\n')
# print(y_test)

DB_DATA_PATH = '../DlibDataChangeLuminace/DB/csv/'
QUERY_DATA_PATH = '../DlibDataChangeLuminace/Query/csv/'

X_train = pd.read_csv(DB_DATA_PATH + 'featurePoint.csv',index_col=0)
y_train = X_train['target'].as_matrix()

X_train = X_train.drop('target', axis=1).as_matrix()

X_test = pd.read_csv(QUERY_DATA_PATH + 'featurePoint.csv',index_col=0)
y_test = X_test['target'].as_matrix()

X_test = X_test.drop('target', axis=1).as_matrix()

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LightGBM parameters
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_logloss'},
        'num_class': 21,
        'learning_rate': 0.1,
        'num_leaves': 32,
        'min_data_in_leaf': 1,
        'num_iteration': 100,
        'verbose': 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_max = np.argmax(y_pred, axis=1)
print(y_pred_max)
print(y_test)

# np.savetxt('lightGBM_Base.csv',y_pred_max,delimiter=';')
accuracy = sum(y_test == y_pred_max) / len(y_test)
print(str(accuracy*100) + '%')