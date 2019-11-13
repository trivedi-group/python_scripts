#Let's import everything we need to get going
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Get the data in!
data = pd.read_csv("path/to/your/data/dat.csv")
print('Data shape', data.shape)

features = data.iloc[:, 2:208]

from scipy.stats import pearsonr

corr = features.corr(method="pearson")
def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]
corr2 = features.corr(method=pearsonr_pval)

fig = plt.figure(figsize=(11,11))
ax = sns.heatmap(corr, vmin=-1, vmax=1, square=False, cmap="autumn")
ax.set_xticklabels(features.columns)
ax.set_yticklabels(features.columns)
plt.show()



