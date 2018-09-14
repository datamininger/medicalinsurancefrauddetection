# coding utf-8

import xgboost as xgb
from Model.ModelBase import ModelBase
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold, KFold
import numpy as np
import pandas as pd
import copy
import pdb
from sklearn.metrics import auc
from utils import compute_fold_result_xgboost, ensemble_submission


class ModelXgboost(ModelBase):

    def offline_validate(self):
        feature_names = list(self.Xtrain.columns)
        cv_params = self.config['cv_params']
        kf = StratifiedKFold(n_splits=cv_params['n_splits'], shuffle=cv_params['shuffle'], random_state=cv_params['random_state'])
        kf.get_n_splits(self.Xtrain[feature_names].values, self.Ytrain['LABEL'].values)

        for train_index, valid_index in kf.split(self.Xtrain[feature_names].values, self.Ytrain['LABEL'].values):
            # 1 分割数据
            X_train, X_valid, y_train, y_valid = self.Xtrain.iloc[train_index], self.Xtrain.iloc[valid_index], self.Ytrain.iloc[train_index], self.Ytrain.iloc[valid_index]
            # 2 转化格式
            xgb_train = xgb.DMatrix(data=X_train[feature_names].values, label=y_train['LABEL'].values, feature_names=feature_names)

            xgb_valid = xgb.DMatrix(data=X_valid[feature_names].values, label=y_valid['LABEL'].values, feature_names=feature_names)
            # 3 训练,
            evals_result = {}
            booster_offline = xgb.train(params=self.config['model']['model_params'],
                                        dtrain=xgb_train,
                                        evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                                        evals_result=evals_result,
                                        **self.config['model']['train_params']
                                        )
            self.booster_offline_list.append(booster_offline)
            self.fold_results.append(compute_fold_result_xgboost(booster_offline, evals_result, feature_names, self.config['norm_feat_imp']))
            self.mean_score_offline = np.mean([fold_result['score_offline'] for fold_result in self.fold_results])
        return

    def online_predict(self):
        feature_names = list(self.Xtrain.columns)
        xgb_test = xgb.DMatrix(self.Xtest[feature_names].values, feature_names=feature_names)
        if self.config['oof']:
            print('online predict by oof ...')
            submissions = []
            for booster in self.booster_offline_list:
                submissions.append(booster.predict(xgb_test, ntree_limit=booster.best_iteration))
            weights = [fold_result['score_offline'] for fold_result in self.fold_results]
            self.submission_online = ensemble_submission(submissions, weights, self.config['ensemble_method'])

        else:
            train_params = copy.deepcopy(self.config['model']['train_params'])
            del train_params["early_stopping_rounds"]

            print('online predict by all train ...')
            xgb_train = xgb.DMatrix(self.Xtrain[feature_names].values, self.Ytrain['LABEL'].values, feature_names=feature_names)
            booster_online = xgb.train(params=self.config['model']['model_params'],
                                       dtrain=xgb_train,
                                       **train_params)
            self.submission_online = booster_online.predict(xgb_test)
        return
