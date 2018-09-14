# coding utf-8

import json
from utils import get_path
import pandas as pd
import numpy as np
import pdb


class ModelBase:
    def __init__(self, Xtrain, Ytrain, Xtest, config):
        """
        :param Xtrain: pd.DataFrame
        :param Ytrain: pd.DataFrame
        :param Xtest: pd.DataFrame
        :param config: dict, 配置字典
        """
        self.Xtrain, self.Ytrain, self.Xtest = Xtrain, Ytrain, Xtest
        self.config = config
        self.booster_offline_list = []
        self.fold_results = []
        self.submission_online = None

        self. mean_score_offline = 0.0

    def save_experiment_result(self, kind='B'):
        ExperimentLog = {'result': {
            'best_iteration': [fold_result['best_iteration'] for fold_result in self.fold_results],
            'score_offline': [fold_result['score_offline'] for fold_result in self.fold_results],
            'score_mean_offline': np.mean([fold_result['score_offline'] for fold_result in self.fold_results]),
            'score_std_offline': np.std([fold_result['score_offline'] for fold_result in self.fold_results]),
            'fold_results': self.fold_results},
            'config': self.config
        }
        print('score is : {}'.format(ExperimentLog['result']['score_offline']))
        print('mean score is : {}'.format(ExperimentLog['result']['score_mean_offline']))
        json.dump(ExperimentLog, open(get_path() + 'Log/ExperimentLog{}.json'.format(self.config['config_name']), 'w'), indent=2)
        # 保存提交文件
        test_id = pd.read_feather(get_path() + 'Data/Feature/test_id.feather')
        test_id['proba'] = self.submission_online
        test_id.to_csv(get_path() + 'Submission/Submission{}_{}.csv'.format(self.config['config_name'], kind), header=False, index=False, sep='\t')
