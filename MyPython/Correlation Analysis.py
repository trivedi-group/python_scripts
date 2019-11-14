#Let's import everything we need to get going
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Get the data in!
data = pd.read_csv("path/to/your/data/lipids.csv")
print('Data shape', data.shape)

features10 = data.iloc[:, 2:22]
features20 = data.iloc[:, 22:32]
features30 = data.iloc[:, 32:42]
features40 = data.iloc[:, 42:52]
features50 = data.iloc[:, 52:62]
features60 = data.iloc[:, 62:72]
features70 = data.iloc[:, 72:82]
features80 = data.iloc[:, 82:92]

from scipy.stats import pearsonr

corr = features20.corr(method="pearson")
def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]
corr2 = features10.corr(method=pearsonr_pval)

ax = sns.heatmap(corr, vmin=-1, vmax=1, square=False, cmap="summer")
ax.set_xticklabels(features.columns)
ax.set_yticklabels(features.columns)
plt.show()
