#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd

from util.plot import *

plot_ranking = True
to_save = True
res = []
RATING_ALGOs = 'ItemMean SlopeOne ItemKNN UserKNN SVD SVDPlusPlus RSTE SocialMF'.split()
# Ranking_ALGOs = 'BPR'.split()
Ranking_ALGOs = 'Rand MostPopular ItemMean SVD SVDPlusPlus BPR NeuMF CFGAN LightGCN'.split()

ALGOs = Ranking_ALGOs if plot_ranking else RATING_ALGOs

filename_pattern = '_top_10-measure-rank' if plot_ranking else '-measure-rating'

for n in range(1):
    for algo_name in ALGOs:
        for on in 'flip_rating'.split():
            for i in range(11):
                filename = f'results/output_{n}/{on}/{algo_name}/{on}_iter_{i}{filename_pattern}[1].csv'
                res.append(pd.read_csv(filename))
df = pd.concat(res)

# In[4]:


columns = df.columns.values.tolist()
columns.remove('value')

# In[5]:


df = df.groupby(columns, as_index=False).mean()

df.loc[df.stage == 'during', 'iteration'] += 11

# In[15]:

if plot_ranking:
    df.metric += '@' + df.k.astype(str)

row_order = 'Gini@10 MIUD@10 nDCG@10 F1@10 Precision@10 Recall@10'.split() if plot_ranking else 'MAE RMSE'.split()
# row_order = 'nDCG@10 Gini@10 F1@10 Precision@10 Recall@10'.split() if plot_ranking else 'MAE RMSE'.split()

g = sns.FacetGrid(
    df, col="metric", hue="algo", sharey=False, hue_order=ALGOs,
    col_wrap=3
    # height=1.7, aspect=5, row_order=row_order,
)
g.map(sns.lineplot, "iteration", "value")
# g.add_legend()
#
last_ax = None
for ax in g.fig.get_axes():
    ax.axvline(x=1, ls='--', c='grey')
    last_ax = ax
#
# y = 0.006 if plot_ranking else 0.73
# offset_image('white', last_ax, x=0, y=y)
# offset_image('black', last_ax, x=10, y=y)
#
# xticks = range(21)
# xlabels = [' '] + list(range(1, 10)) + [' '] + list(range(1, 11))
# last_ax.set_xticks(xticks)
# last_ax.set_xticklabels(xlabels, ha='center')
#
# g.add_legend()
# lgd = plt.legend(
#     loc='lower center',
#     bbox_to_anchor=(0.5, -0.85 if plot_ranking else -1),
#     ncol=5 if plot_ranking else 4,
#     fancybox=False, shadow=False
# )
lgd = plt.legend(
    loc='lower center',
    bbox_to_anchor=(-0.8, -0.45 if plot_ranking else -1),
    ncol=5 if plot_ranking else 4,
    fancybox=False, shadow=False
)

os.makedirs('results/images', exist_ok=True)
res_filename = 'ranking' if plot_ranking else 'rating'
if to_save:
    # plt.savefig(f'results/images/{res_filename}-result.svg')
    plt.savefig(f'results/images/{res_filename}-result.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # df.to_csv(f'results/{res_filename}-result.csv', index=False)
# plt.tight_layout()
plt.show()
