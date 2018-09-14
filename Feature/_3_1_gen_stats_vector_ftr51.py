from utils import IsAbsense, SaveFeature, get_path, ReadData, stats_FTR51_by_size
import multiprocessing
import json
import pandas as pd
import pdb
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
from sklearn.externals import joblib


def gen_stats_vector_ftr51(stats_name, size='7d', non_zero=False):
    """
    :param stats_name: str,对药品数量进行统计的名字
    :param size: str, 统计的时间粒度 1d, 4d, 7d, 15d, 30d, 45d
    :param non_zero: bool, 统计是否非0
    :return:
    """
    assert stats_name in ['sum', 'sum_ratio', 'max', 'max_ratio', 'mean', 'std']
    mask = (stats_name in ['sum', 'sum_ratio', 'max', 'max_ratio']) & non_zero
    assert not mask

    matrix_name = '{}_vector_ftr51_by_{}_{}'.format(stats_name, size, non_zero)
    # 0 读取数据
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
    train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 计算统计字典
    print('1 computing stats dict {}'.format(size))
    ftr51_stats_dict_df = train_test_data[['PERSONID', 'CREATETIME', 'FTR51']].groupby('PERSONID').apply(
        lambda df_person: stats_FTR51_by_size(df_person, stats_name, size, non_zero)).to_frame('stats_dict').reset_index()
    train_test_id = train_test_id.merge(ftr51_stats_dict_df, on=['PERSONID'], how='left')
    v = DictVectorizer()
    # 计算统计向量
    print('2 computing stats vector'.format(size))
    ftr51_stats_sparse_matrix = v.fit_transform(train_test_id['stats_dict'].values)
    joblib.dump(v, 'v_{}_{}.m'.format(stats_name, size))
    sparse.save_npz(get_path() + 'Data/Feature/{}.npz'.format(matrix_name), ftr51_stats_sparse_matrix)

    return matrix_name, 'gen_stats_vector_ftr51("{}", "{}", {})'.format(stats_name, size, non_zero)

gen_stats_vector_ftr51('mean')

"""
if __name__ == '__main__':

    batch_name = '20180801_am_2'
    # stats_name = ['sum', 'sum_ratio', 'max', 'max_ratio', 'mean', 'std']
    stats_name = ['mean', 'std', 'sum']

    pool = multiprocessing.Pool(processes=3)
    func = gen_stats_vector_ftr51
    func_name = 'gen_stats_vector_ftr51_30d'

    feature_hist_add = {}
    for result in pool.imap_unordered(func, stats_name):
        feature_name, gen_method_string = result[0], result[1]
        feature_hist_add.setdefault(feature_name, gen_method_string)
    # 读取历史文件
    path_feature_hist = get_path() + 'FeatureGenHistory/{}.json'.format(func_name)
    feature_hist_total = json.load(open(path_feature_hist))
    # 更新字典
    feature_hist_total.setdefault(batch_name, feature_hist_add)
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)
"""