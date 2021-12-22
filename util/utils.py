import time

import pandas as pd
from surprise import Prediction


def df_to_prediction(df: pd.DataFrame):
    """

    :param df: assume the columns are: 'uid iid r_ui est *'
    :return:
    """
    return df.apply(lambda r: Prediction(int(r.uid), int(r.iid), r.r_ui, r.est, ''), axis=1)


def readable_time(timestamp: float = time.time()):
    return time.ctime(int(timestamp))
