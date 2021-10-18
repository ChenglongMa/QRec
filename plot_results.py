import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from util.config import Config

result_dir = './results'
# ALGOs = 'APR BUIR CDAE CFGAN DiffNet NGCF'.split()  # MostPopular Rand BPR SVD SVD++ SlopeOne
ALGOs = 'SlopeOne UserKNN ItemKNN BasicMF NeuMF SVD SVD++'.split()  # MostPopular Rand BPR SVD SVD++ SlopeOne

if __name__ == '__main__':

    evaluation_results = []
    for algo in ALGOs:
        config = Config(f"./config/{algo}.conf")
        algo_name = config['model.name']
        test_on = 'during'

        for i in range(11):
            eval_res = pd.read_csv(f'{result_dir}/{algo_name}iter_{test_on}_{i}-measure[1].txt', sep=':',
                                   names='metric value'.split())
            eval_res['algo'] = algo_name
            eval_res['iteration'] = i
            eval_res['k'] = 10
            evaluation_results.append(eval_res)
            # ranking:
            # evaluation_results.append(pd.read_csv(f'{result_dir}/{algo_name}/iter_{test_on}_{i}-measure[1].csv'))

    eval_df = pd.concat(evaluation_results).reset_index(drop=True)
    for m in eval_df.metric.unique():
        plt.figure()
        sns.lineplot(data=eval_df[eval_df.metric == m], x='iteration', y='value', hue='algo', marker='o')
        plt.title(m)
    plt.show()
