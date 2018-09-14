from utils import IsAbsense, SaveFeature, get_path, ReadData, compute_cat_count_dict_from_ftr51s, IsDifferentDistribution, stats_by_oob_dict
import multiprocessing
import json
import pandas as pd
import pdb
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
import numpy as np


def ftr51s2cat_count_dict(df_person, kinds):
    ftr51s = ','.join(list(df_person['FTR51'].values))
    count_dict = compute_cat_count_dict_from_ftr51s(ftr51s, kinds)
    return count_dict


def subtract_dict(dict1, dict2):
    # pdb.set_trace()
    dict3 = {key: (dict1[key] - dict2[key]) for key in dict2.keys()}
    return dict3


def division_dict(dict1, dict2):
    dict3 = {key: dict2[key] / (dict1[key] + 0.000000001) for key in dict2.keys()}
    return dict3


def mean_non_zero(arr):
    mask = (arr != 0)
    mean_value = pd.Series(arr[mask]).mean()
    return mean_value


def repair_fraud_dict_person(x):
    if pd.isna(x['fraud_dict_person']):
        return {key: 0 for key in x['count_dict_person']}
    else:
        # 这样 fraud count 对齐了
        return {key: x['fraud_dict_person'].setdefault(key, 0) for key in x['count_dict_person']}


def gen_fraud_ratio_feature(kinds='E', stats_name='fraud_ratio_mean_weight'):
    """
    计算一个人所有的cat, 计算cat oob 的count， fraud, 例如某欺诈用户如果B1一次记录出现两次，则B1 fraud +2, count +2,
    利用count， fraud 计算统计值
    :param kinds: str, 目标编码的 字符, 可以是 ABCDE 或其组合
    :return:
    """
    feature_name = '{}_{}'.format(stats_name, kinds)
    print('computing feature {}'.format(feature_name))
    # 0 读取数据
    train_id, test_id, train_data, test_data, Ytrain = ReadData(Ytrain=True, sort_by_time=True)
    train_id['LABEL'] = Ytrain['LABEL']
    train_data = train_data.merge(train_id, on=['PERSONID'], how='left')
    train_id = train_id.drop(['LABEL'], axis=1)
    # 1 个人计数
    df_cat_person_count = train_data[['PERSONID', 'FTR51']].groupby('PERSONID').apply(lambda df_person: ftr51s2cat_count_dict(df_person, kinds)).to_frame(
        'count_dict_person').reset_index()
    train_id = train_id.merge(df_cat_person_count, on=['PERSONID'], how='left')
    # 2 个人欺诈计数
    mask = train_data['LABEL'] == 1
    df_cat_person_fraud = train_data[mask][['PERSONID', 'FTR51']].groupby('PERSONID').apply(lambda df_person: ftr51s2cat_count_dict(df_person, kinds)).to_frame(
        'fraud_dict_person').reset_index()
    train_id = train_id.merge(df_cat_person_fraud, on=['PERSONID'], how='left')
    # ---------------------------------------- 好深的bug
    # 这样一来，如果非欺诈人员就没有个人欺诈记录，值全部为0,
    train_id['fraud_dict_person'] = train_id[['count_dict_person', 'fraud_dict_person']].apply(lambda x: repair_fraud_dict_person(x), axis=1)
    # ---------------------------------------  好深的bug
    # 3 所有计数
    ftr51s_all = ','.join(list(train_data['FTR51'].values))
    count_dict_all = compute_cat_count_dict_from_ftr51s(ftr51s_all, kinds)
    # 4 所有欺诈
    ftr51s_all_fraud = ','.join(list(train_data[mask]['FTR51'].values))
    fraud_dict_all = compute_cat_count_dict_from_ftr51s(ftr51s_all_fraud, kinds)
    fraud_dict_all = {key: fraud_dict_all.setdefault(key, 0) for key in count_dict_all.keys()}
    # 5 赋值
    train_id['count_dict_all'] = [count_dict_all for _ in range(train_id.shape[0])]
    train_id['fraud_dict_all'] = [fraud_dict_all for _ in range(train_id.shape[0])]

    # 6 oob dict
    train_id['count_dict_oob'] = train_id[['count_dict_all', 'count_dict_person']].apply(lambda s: subtract_dict(s['count_dict_all'], s['count_dict_person']),
                                                                                         axis=1)

    train_id['fraud_dict_oob'] = train_id[['fraud_dict_all', 'fraud_dict_person']].apply(lambda s: subtract_dict(s['fraud_dict_all'], s['fraud_dict_person']),
                                                                                         axis=1)

    count_dict_person_test = test_data[['PERSONID', 'FTR51']].groupby('PERSONID').apply(lambda df_person: ftr51s2cat_count_dict(df_person, kinds)).to_frame(
        'count_dict_person').reset_index()
    test_id = test_id.merge(count_dict_person_test, on=['PERSONID'], how='left')
    test_id['fraud_dict_oob'] = [fraud_dict_all for _ in range(test_id.shape[0])]
    test_id['count_dict_oob'] = [count_dict_all for _ in range(test_id.shape[0])]

    test_id['count_dict_oob'] = test_id.apply(lambda x: {key: x['count_dict_oob'].setdefault(key, 0) for key in x['count_dict_person'].keys()}, axis=1)
    test_id['fraud_dict_oob'] = test_id.apply(lambda x: {key: x['fraud_dict_oob'].setdefault(key, 0) for key in x['count_dict_person'].keys()}, axis=1)

    # 统计计算特征

    train_id[feature_name] = train_id.apply(lambda s: stats_by_oob_dict(s, stats_name), axis=1)
    test_id[feature_name] = test_id.apply(lambda s: stats_by_oob_dict(s, stats_name), axis=1)
    SaveFeature(train_id, test_id, feature_name)
    IsDifferentDistribution(feature_name)




gen_fraud_ratio_feature()
