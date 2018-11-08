#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
sns.boxplot(x="target", y="right_Eyebrow_height",
            data=df_X_train.sort_values('target'))

#%%
sns.boxplot(x="target", y="mouse_width", data=df_X_train.sort_values('target'))

#%%
sns.jointplot('nose_x3', 'nose_y3', data=df_X_train)

#%%
fig, ax = plt.subplots(figsize=(10, 10))
df_X_train_corr = df_X_train.corr()
sns.heatmap(df_X_train_corr, square=True, vmax=1, vmin=-1, center=0)
