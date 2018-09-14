# coding utf-8

import lightgbm as lgb
from Model.ModelBase import ModelBase
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np
import copy
import pdb
import pandas as pd
from utils import compute_fold_result_lightgbm, ensemble_submission


class ModelLightgbm(ModelBase):

    def offline_validate(self):
        feature_names = list(self.Xtrain.columns)
        cv_params = self.config['cv_params']
        kf = KFold(n_splits=cv_params['n_splits'], shuffle=cv_params['shuffle'], random_state=cv_params['random_state'])
        kf.get_n_splits(self.Xtrain[feature_names].values, self.Ytrain['LABEL'].values)
        for train_index, valid_index in kf.split(self.Xtrain[feature_names].values, self.Ytrain['LABEL'].values):
            # 1 分割数据
            X_train, X_valid, y_train, y_valid = self.Xtrain.iloc[train_index], self.Xtrain.iloc[valid_index], self.Ytrain.iloc[train_index], self.Ytrain.iloc[valid_index]
            # 2 转化格式
            lgb_train = lgb.Dataset(data=X_train[feature_names].values,
                                    label=y_train['LABEL'].values)
            lgb_valid = lgb.Dataset(data=X_valid[feature_names].values,
                                    label=y_valid['LABEL'].values)
            # 3 训练,
            evals_result = {}
            booster_offline = lgb.train(params=self.config['model']['model_params'],
                                        train_set=lgb_train,
                                        valid_sets=[lgb_train, lgb_valid],
                                        valid_names=['train', 'valid'],
                                        evals_result=evals_result,
                                        feature_name=feature_names,
                                        **self.config['model']['train_params'])
            # 4 得到结果
            self.booster_offline_list.append(booster_offline)
            self.fold_results.append(compute_fold_result_lightgbm(booster_offline, evals_result, feature_names, self.config['norm_feat_imp']))
            self.mean_score_offline = np.mean([fold_result['score_offline'] for fold_result in self.fold_results])

        return

    def online_predict(self):
        feature_names = list(self.Xtrain.columns)
        if self.config['oof']:
            submissions = []
            for booster in self.booster_offline_list:
                submissions.append(booster.predict(self.Xtest[feature_names].values, num_iteration=booster.best_iteration))
            weights = [fold_result['score_offline'] for fold_result in self.fold_results]
            self.submission_online = ensemble_submission(submissions, weights, self.config['ensemble_method'])
        else:
            train_params = copy.deepcopy(self.config['model']['train_params'])
            del train_params["early_stopping_rounds"]
            lgb_train = lgb.Dataset(data=self.Xtrain[feature_names].values,
                                    label=self.Ytrain['LABEL'].values)
            booster_online = lgb.train(params=self.config['model']['model_params'],
                                       train_set=lgb_train,
                                       feature_name=feature_names,
                                       **train_params)
            self.submission_online = booster_online.predict(self.Xtest[feature_names].values)
        return

