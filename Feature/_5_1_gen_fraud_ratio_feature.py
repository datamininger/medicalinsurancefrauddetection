from utils import IsAbsense, SaveFeature, get_path, ReadData, compute_cat_count_dict_from_ftr51s, IsDifferentDistribution
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


def gen_fraud_ratio_feature(kinds='B'):
    """
    :param kinds: str, 目标编码的 字符, 可以是 ABCDE 或其组合
    :return:
    """
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
    # 7 cat fraud  ratio dict
    # train
    train_id['cat_fraud_ratio_dict_oob'] = train_id[['count_dict_oob', 'fraud_dict_oob']].apply(
        lambda s: division_dict(s['count_dict_oob'], s['fraud_dict_oob']),
        axis=1)
    # test
    cat_fraud_ratio_dict_all = division_dict(count_dict_all, fraud_dict_all)
    test_id['cat_fraud_ratio_dict_oob'] = [cat_fraud_ratio_dict_all for _ in range(test_id.shape[0])]
    count_dict_person_test = test_data[['PERSONID', 'FTR51']].groupby('PERSONID').apply(lambda df_person: ftr51s2cat_count_dict(df_person, kinds)).to_frame(
        'count_dict_person').reset_index()
    test_id = test_id.merge(count_dict_person_test, on=['PERSONID'], how='left')
    test_id['cat_fraud_ratio_dict_oob'] = test_id.apply(
        lambda x: {key: x['cat_fraud_ratio_dict_oob'].setdefault(key, 0) for key in x['count_dict_person'].keys()}, axis=1)

    # 利用cat的欺诈比生成个人的特征
    # 8 max_fraud_ratio 特征
    train_id['max_fraud_ratio'] = train_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).max())
    test_id['max_fraud_ratio'] = test_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).max())

    #  9 sum_fraud_ratio 特征
    train_id['sum_fraud_ratio'] = train_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).sum())
    test_id['sum_fraud_ratio'] = test_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).sum())

    # 10  mean_fraud_ratio 特征
    train_id['mean_fraud_ratio'] = train_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).mean())
    test_id['mean_fraud_ratio'] = test_id['cat_fraud_ratio_dict_oob'].map(lambda fraud_ratio_dict: pd.Series(fraud_ratio_dict).mean())

    # 11 保存特征, 查看分布
    for feat in ['max_fraud_ratio', 'sum_fraud_ratio', 'mean_fraud_ratio']:
        SaveFeature(train_id, test_id, feat)
        IsDifferentDistribution(feat)


gen_fraud_ratio_feature()
