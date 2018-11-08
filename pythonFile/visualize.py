#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
DB_PATH = '/Users/reo/.ghq/github.com/reo11/TMUProjectExpB1/DlibDataChangeLuminace/DB/csv/'
os.chdir(DB_PATH)
os.getcwd()

#%%
df_X_train = pd.read_csv('features_basic_divFaceWidth.csv', index_col=0)
# X_test = pd.read_csv(
#     '/Users/reo/.ghq/github.com/reo11/TMUProjectExpB1/DlibDataChangeLuminace/Query/csv/featurePoint_nosevec_normalize.csv', index_col=0)
# y_train = df_X_train['target'].as_matrix()
df_X_train.head()

#%%
df_X_train.describe()

#%%
for i in range(1, len(df_X_train.columns)):
    a = sns.boxplot(x=df_X_train.columns[0], y=df_X_train.columns[i],
            data=df_X_train.sort_values('target'))
    plt.figure()

#%%
fig, ax = plt.subplots(figsize=(10, 10))
df_X_train_corr = df_X_train.corr()
sns.heatmap(df_X_train_corr, square=True, vmax=1, vmin=-1, center=0)

#%%
sns.jointplot(df_X_train.columns[0],
              df_X_train.columns[1], df_X_train, kind="kde")

#%%
sns.pairplot(df_X_train, hue='target')
