from utils import IsAbsense, SaveFeature, get_path, ReadData, compute_cat_count_dict_from_ftr51s
import multiprocessing
import json
import pandas as pd
import pdb
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse


def gen_cat_vector_from_ftr51(kinds):
    """
    为train_test_data
    :param kinds: str, A, D, AB
    :return:
    """
    print('compute {} vector for train_test_data'.format(kinds))
    matrix_name = '{}_vector_from_ftr51'.format(kinds)
    # 0 读取数据
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 计数字典字典
    train_test_data['cat_count_dict'] = train_test_data['FTR51'].map(lambda ftr51s: compute_cat_count_dict_from_ftr51s(ftr51s, kinds))
    # 对齐
    v = DictVectorizer()
    # 计算统计向量
    cat_sparse_matrix = v.fit_transform(train_test_data['cat_count_dict'].values)
    sparse.save_npz(get_path() + 'Data/Feature/{}.npz'.format(matrix_name), cat_sparse_matrix)

    return matrix_name, 'gen_cat_vector_from_ftr51("{}")'.format(kinds)


gen_cat_vector_from_ftr51('AC')
"""

if __name__ == '__main__':

    batch_name = '20180801_am_2'
    # stats_name = ['sum', 'sum_ratio', 'max', 'max_ratio', 'mean', 'std']
    stats_name = ['mean', 'std']

    pool = multiprocessing.Pool(processes=6)
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