import os
import time

from QRec import QRec
from util.cmd import get_argv
from util.config import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
n = get_argv(1, 0)
iteration = get_argv(2, 0)
algo = get_argv(3)
test_on = get_argv(4, 'flip_rating')
ranking = get_argv(5, False)
top_n = get_argv(6, 10)
n_normal_users = get_argv(7, 0)

if __name__ == '__main__':
    s = time.time()
    dataset_type = 'ranking' if ranking else 'rating'
    # C:\Users\ChenglongMa\Documents\Projects\DataScience\interaction_with_recsys\data\ranking_0\n_normal_users_0\uirt\scaled\during_test_iter_0.csv
    # C:\Users\ChenglongMa\Documents\Projects\DataScience\HumanNeedSimulation\data\flip_rating\train_iter_0.csv
    data_dir = f'C:/Users/ChenglongMa/Documents/Projects/DataScience/HumanNeedSimulation/data'

    config = Config(f"./config/{algo}.conf")

    algo_name = config['model.name']

    config['ratings'] = f'{data_dir}/{test_on}/train_iter_{iteration}.csv'
    config['item.ranking'] = f'on -topN {top_n}' if ranking else 'off -topN -1'
    config['iteration'] = iteration
    config['testOn'] = test_on
    result_dir = f'./results/output_{n}/{test_on}/'
    config['output.setup'] = f'on -dir {result_dir}'
    sub_result_dir = f'{result_dir}/{algo_name}'
    os.makedirs(sub_result_dir, exist_ok=True)
    config['evaluation.setup'] = f'-testSet {data_dir}/{test_on}/test_iter_{iteration}.csv'
    rec = QRec(config)
    rec.execute()

    e = time.time()
    seconds = e - s
    print(f"Single Task Run time: {seconds:.2f} s = {seconds // 60} mins")
