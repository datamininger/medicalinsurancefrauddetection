# coding: utf-8
import numpy as np
import pandas as pd
from utils import get_path


def to_feather():
    """
    将原有数据转换成feather格式便于快速读取
    假设只有一个test，不区分A、B， 待B榜时，将A、B榜数据合并形成新的test
    :return:
    """
    path = get_path() + 'Data/RawData/'
    train = pd.read_table(path + 'train.tsv')
    train_id = pd.read_table(path + 'train_id.tsv')
    test = pd.read_table(path + 'test_B.tsv')

    train['CREATETIME'] = pd.to_datetime(train['CREATETIME'])
    test['CREATETIME'] = pd.to_datetime(test['CREATETIME'])

    train.to_feather(path + 'train.feather')
    test.to_feather(path + 'test.feather')
    train_id.to_feather(path + 'train_id.feather')
    return train, train_id, test


def gen_train_test_id_Ytrain():
    path = get_path() + 'Data/'
    train_id = pd.read_feather(path + 'RawData/train_id.feather')
    test_data = pd.read_feather(path + 'RawData/test.feather')

    # Ytrain, train_id
    train_id[['LABEL']].to_feather(path + 'Feature/Ytrain.feather')
    train_id[['PERSONID']].to_feather(path + 'Feature/train_id.feather')
    # test id
    test_B_id = pd.DataFrame(test_data['PERSONID'].unique(), columns=['PERSONID'])
    test_B_id[['PERSONID']].to_feather(path + 'Feature/test_id.feather')

to_feather()
gen_train_test_id_Ytrain()
