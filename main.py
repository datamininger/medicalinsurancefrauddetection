# coding: utf-8
import json
from utils import CombineFeature, ReadExperimentLog, FilterFeature, IsDifferentDistribution, AddBlackFeature, ReadData
from Model.InitModel import InitModel
import pdb
import xgboost as xgb
import numpy as np

def gen_config_Lightgbm():
    config = {'config_name': 1,
              'note': '2018-07-17',

              'feature_names': [],
              "model": {
                  "name": "Lightgbm",
                  "model_params": {
                      'objective': 'binary',
                      'boosting_type': 'gbdt',
                      "learning_rate": 0.01,
                      'metrics': 'auc',
                      'scale_pos_weight': 1,
                      'num_leaves': 31,
                      'max_depth': 4,
                      'min_child_weight': 0,
                      'reg_lambda': 0.0,
                      'min_split_gain': 0,
                      'subsample_freq': 1,
                      'subsample': 1,
                      'colsample_bytree': 0.6,
                      'num_threads': 20,
                      "verbose": 1,
                      "seed": 2018,
                  },
                  "train_params": {
                      "num_boost_round": 60000,
                      "early_stopping_rounds": 1000,
                  },
              },
              'cv_params': {'shuffle': True,
                            'random_state': 100,
                            'n_splits': 5,
                            },
              'save_experiment_result': True,
              'oof': True,
              "ensemble_method": 'mean',
              'norm_feat_imp': True}
    return config


def gen_config_Xgboost():
    config = {'config_name': 1,
              'note': '2018-07-18',
              'feature_names': [],

              # ["count_max_ftr51_in_month4", "count_max_ftr51_in_month9", "count_max_ftr51_in_month10", "count_max_ftr51_in_month11"],

              "model": {
                  "name": "Xgboost",
                  "model_params": {
                      'objective': 'binary:logistic',
                      #  'objective': 'rank:pairwise',
                      'learning_rate': 0.1,
                      'eval_metric': 'auc',
                      'scale_pos_weight': 1,
                      # 'num_leaves': 31,
                      'max_depth': 4,
                      'gamma': 0,
                      'min_child_weight': 1,
                      'lambda': 1,
                      'alpha': 0,
                      'colsample_bytree': 0.7,
                      'subsample': 0.9,
                      'colsample_bylevel': 0.7,
                      'nthread': 20,
                      'silent': True,
                      "verbose": 0,
                      "seed": 2018,
                  },
                  "train_params": {
                      "num_boost_round": 500,
                      "early_stopping_rounds": 60,
                  },
              },
              'cv_params': {'shuffle': True,
                            'random_state': 100,
                            'n_splits': 5,
                            },
              'save_experiment_result': True,
              'oof': True,
              "ensemble_method": 'mean',
              'norm_feat_imp': True}
    return config


def gen_config_catboost():
    return


def gen_config_base(model_name):
    """
    :param model_name:
    :return:
    """
    if model_name == 'Xgboost':
        return gen_config_Xgboost()
    elif model_name == 'Lightgbm':
        return gen_config_Lightgbm()


def gen_config_feature(model_name, log_name_set, new_feature_hist_path, feature_batch_name_set, filter_black, add_new):
    """
    保留上一次实验的特征, 添加特征历史文件中的某批次特征, 选择模型基本参数, 过滤特征, 形成配置
    :param model_name: str
    :param log_name_set: int or str, 用于读取实验记录中的特征
    :param new_feature_hist_path: str, 新特征历史文件地址
    :param feature_batch_name_set: str, 如'20180102am'
    :param model_name: str, 模型名字,,如Lightgbm, Xgboost
    :return:
    """
    # 1 读取基础配置
    base_config = gen_config_base(model_name)
    feature_names = base_config['feature_names']

    # 2 读取旧特征
    old_features = []
    for log_name in log_name_set:
        old_features += ReadExperimentLog(log_name)['config']['feature_names']

    # 3 根据批次, 添加新测试的特征
    new_features = []
    if add_new:
        for feature_batch_name in feature_batch_name_set:
            new_feature_dict = json.load(open(new_feature_hist_path))[feature_batch_name]
            for new_feature in new_feature_dict.keys():
                new_features.append(new_feature)

    feature_names = feature_names + old_features + new_features

    # 4 唯一和排序
    # feature_names = list(set(feature_names))
    # feature_names.sort()

    # 5 特征黑名单过滤
    if filter_black:
        feature_names = FilterFeature(feature_names)
    # 6 更新特征名并返回
    base_config['feature_names'] = feature_names

    # 7 不在黑名单的新特征, 有时会重复计算特征
    print('the size of feature name is ', len(feature_names))

    new_feature_list = [new_feature for new_feature in new_features if new_feature in feature_names]
    old_feature_list = [old_feature for old_feature in old_features if old_feature in feature_names]
    return base_config, new_feature_list, old_feature_list


def run(config):
    """
    :param config: dict, 配置字典
    :return:
    """
    # 1 根据配置合并特征
    Xtrain, Ytrain, Xtest = CombineFeature(config['feature_names'])

    # ------------------------
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    Xtrain['PERSONID'] = train_id['PERSONID']
    Ytrain['PERSONID'] = train_id['PERSONID']
    Xtest['PERSONID'] = test_id['PERSONID']
    Xtrain.to_csv('Xtrain_xiao.csv', index=False)
    Ytrain.to_csv('Ytrain_xiao.csv', index=False)
    Xtest.to_csv('Xtest_xiao.csv', index=False)

    Xtrain.drop(['PERSONID'], axis=1, inplace=True)
    Ytrain.drop(['PERSONID'], axis=1, inplace=True)
    Xtest.drop(['PERSONID'], axis=1, inplace=True)
    # ------------------------

    # 2 根据配置初始化模型
    model = InitModel(Xtrain, Ytrain, Xtest, config)
    # 3 线下验证
    model.offline_validate()
    # 4 线上预测
    model.online_predict()
    # 保存实验结果
    if config['save_experiment_result']:
        model.save_experiment_result()
    # 6 返回线下验证分数以及显示预测结果

    # 保存模型
    for i, booster in enumerate(model.booster_offline_list):
        booster.save_model('xgb{}.m'.format(i))

    # 连接模型预测
    feature_names = list(Xtest.columns)
    xgb_test = xgb.DMatrix(Xtest[feature_names].values, feature_names=feature_names)

    submission_list = []
    for i, best_iter in enumerate([161, 292, 160, 246, 269]):
        load_model = xgb.Booster(model_file='xgb{}.m'.format(i))
        submission_list.append(load_model.predict(xgb_test, ntree_limit=best_iter))
    submission = np.mean(submission_list, axis=0)
    print(np.sum(np.abs(model.submission_online - submission)))

    return model.mean_score_offline, model.submission_online, model.fold_results


def select_feature(log_name, n, log_1):
    feature_names = []
    log = ReadExperimentLog(log_name)

    fold_results = log['result']['fold_results']
    for fold_result in fold_results:
        feats = [feat_tuple[0] for feat_tuple in sorted(fold_result['feature_importance_dict'].items(), key=lambda item: item[1])]
        feature_names += feats[-n:]

    feature_names = list(set(feature_names))
    print('The number of feature_names is ', len(feature_names))
    config = log['config']
    config['feature_names'] = feature_names
    config['config_name'] = log_1
    run(config)
    return feature_names




def main():
    # 用于特征测试以及特征选择
    # 0
    config_name = '57'
    config_note = 'gen_stats_value_ftr51'
    plot_new_feature = True
    plot_old_feature = True
    # 1 实验启动要素
    experiment_params = {
        'model_name': 'Xgboost',
        'log_name_set': [43],
        'new_feature_hist_path': 'FeatureGenHistory/gen_stats_cost_diff_7d.json',
        'feature_batch_name_set': ['20180730_am_{}'.format(i) for i in range(1,3)],
        'filter_black': False,
        'add_new':False}
    test_config, new_feature_list, old_feature_list = gen_config_feature(**experiment_params)
    # 3 配置名字
    test_config['config_name'] = config_name
    test_config['note'] = config_note
    # 4 运行该配置
    fold_results = run(test_config)[2]
    # pdb.set_trace()
    # 5 判断新测试的特征是否黑特征，添加到黑名单
    if plot_new_feature:
        for feature in new_feature_list:
            if feature in fold_results[0]['feature_importance_dict'].keys():
                try:
                    print('feature importance is :', [fold_results[i]['feature_importance_dict'][feature] for i in range(5)])
                    if IsDifferentDistribution(feature):
                        AddBlackFeature([feature])
                except:
                    print('error')
    # 6
    if plot_old_feature:
        for feature in old_feature_list:
            if feature in fold_results[0]['feature_importance_dict'].keys():
                try:
                    print('feature importance is :', [fold_results[i]['feature_importance_dict'][feature] for i in range(5)])
                    if IsDifferentDistribution(feature):
                        AddBlackFeature([feature])
                except:
                    print('error')

main()

