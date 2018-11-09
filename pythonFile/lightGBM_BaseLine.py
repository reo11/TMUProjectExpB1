import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
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

X_train = pd.read_csv(DB_DATA_PATH + 'features_rel_dist.csv', index_col=0)
y_train = X_train['target'].as_matrix()

X_train = X_train.drop('target', axis=1).as_matrix()
df = pd.read_csv(DB_DATA_PATH + 'features_rel_dist.csv', index_col=0)
X_test = pd.read_csv(QUERY_DATA_PATH + 'features_rel_dist.csv', index_col=0)
y_test = X_test['target'].as_matrix()

X_test = X_test.drop('target', axis=1).as_matrix()

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LightGBM parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': {'multi_error'},
    'num_class': 21,
    'learning_rate': 0.01,
    'num_leaves': 32,
    'min_data_in_leaf': 2,
    'num_iteration': 200,
    'verbose': 20
}

# train
cols_to_drop = ['target']
cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=50,
                valid_sets=lgb_eval,
                early_stopping_rounds=100)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_max = np.argmax(y_pred, axis=1)
print(y_pred_max)
print(y_test)

feature_importance = pd.DataFrame(sorted(
    zip(gbm.feature_importance(), cols_to_fit)), columns=['Value', 'Feature'])

plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(
    by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('figure.png')

# np.savetxt('lightGBM_Base.csv',y_pred_max,delimiter=';')
accuracy = sum(y_test == y_pred_max) / len(y_test)
print(str(accuracy*100) + '%')
