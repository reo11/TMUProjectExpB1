#%%
import numpy as np
import pandas as pd
import seaborn as sns

DB_DATA_PATH = '../DlibDataChangeLuminace/DB/csv/'
QUERY_DATA_PATH = '../DlibDataChangeLuminace/Query/csv/'

X_train = pd.read_csv(DB_DATA_PATH + 'featurePoint_nosevec_normalize.csv',index_col=0)
X_test = pd.read_csv(QUERY_DATA_PATH + 'featurePoint_nosevec_normalize.csv',index_col=0)

y_train = X_train['target'].as_matrix()

sns.distplot(y_train, kde=False, rug=False, bins=10)