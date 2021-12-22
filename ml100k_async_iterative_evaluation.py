import subprocess
import time
from collections import defaultdict
import os.path

import numpy as np
import pandas as pd

from util.config import Config
from util.utils import readable_time

if __name__ == '__main__':

    async_run = True
    enable_rating_task = True
    enable_ranking_task = True
    s = time.time()
    cmds = []
    # eval_filenames = defaultdict(list)
    # n iteration algo test_on ranking top_n
    # for n in range(1):

    for i in range(11):
        for on in 'flip_rating'.split():
            #     for n_normal_users in [25]:
            output_dir = f'./results/output_0/{on}'
            if enable_rating_task:
                # Cite count:
                # EE: 83
                # SocialFD 29 x
                # SocialMF 760
                # SREE x
                # RSTE 976
                # CUNE-MF
                for algo in 'SlopeOne SocialMF RSTE SVD ItemKNN UserKNN ItemMean SVD++'.split():  # TODO: change algorithm list
                    # algo_name = Config(f"./config/{algo}.conf")['model.name']
                    is_rank = False
                    cmds.append(f'python mlrun.py 0 {i} {algo} {on} {is_rank} -1 0')
                    # eval_filenames[n].append(f'{output_dir}/{algo_name}/{on}_iter_{i}-measure-rating[1].csv')
            if enable_ranking_task:
                # for algo in 'Rand'.split():
                for algo in 'BPR Rand MostPopular ItemMean SVD SVD++ LightGCN CFGAN NeuMF'.split():
                    # for algo in 'SVD++ SVD Rand MostPopular ItemMean'.split():
                    # sync algos: CFGAN LightGCN CDAE NeuMF CFGAN
                    # Expected: WRMF CDAE LightGCN SVD++ SVD Rand MostPopular NeuMF IRGAN IF_BPR NGCF CFGAN BUIR SERec
                    # Unexpected: BPR DMF APR DiffNet SGL
                    # Bugs: CoFactor
                    # algo_name = Config(f"./config/{algo}.conf")['model.name']
                    is_rank = True
                    top_n = 10
                    cmds.append(f'python mlrun.py 0 {i} {algo} {on} {is_rank} {top_n} 0')
                    # eval_filenames[n].append(
                    #     f'{output_dir}/{algo_name}/{on}_iter_{i}_top_{top_n}-measure-rank[1].csv'
                    # )

    print(f'{len(cmds)} tasks ready to run:')

    if async_run:
        jobs = np.array(cmds).reshape((-1, min(17, len(cmds))))
        for job in jobs:
            print('To run:')
            print(job)
            processes = [subprocess.Popen(program) for program in job]
            for process in processes:
                process.wait()
    else:
        for cmd in cmds:
            os.system(cmd)

    # for n, filenames in eval_filenames.items():
    #     eval_results = pd.concat([pd.read_csv(f) for f in filenames if os.path.isfile(f)])
    #     eval_results.to_csv(f'./results/output_{n}/evaluation.csv', index=False)

    e = time.time()
    print('Done!')
    seconds = e - s
    print(f"Total Run time: {seconds:.2f} s = {seconds // 60} mins")
    print(readable_time())
