from utils import IsAbsense, SaveFeature, get_path, ReadData, stats_FTR51_by_size
import multiprocessing
import json
import pandas as pd
import pdb
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def gen_decomposition_stats_vector_from_cat_vector(stats_name, kinds, size='30d', decomp_method='lda', n_components=20):
    """
    :param stats_name: str,对药品数量进行统计的名字
    :param size: str, 统计的时间粒度 1d, 4d, 7d, 15d, 30d, 45d
    :param decomp_method: str, 分解方法
    :param n_components: int , 分解之后的维度
    :return:
    """
    assert decomp_method in ['svd', 'nmf', 'lda']

    stats_matrix_name = '{}_{}_vector_by_{}'.format(stats_name, kinds, size)
    # 0 读取数据
    stats_sparse_matrix = sparse.load_npz(get_path() + 'Data/Feature/{}.npz'.format(stats_matrix_name)).toarray()

    print(0)
    if decomp_method == 'svd':
        print(' svd decomposition...')
        svd = TruncatedSVD(n_components=n_components, n_iter=50, random_state=42)
        stats_matrix_decomp = svd.fit_transform(stats_sparse_matrix)

    if decomp_method == 'nmf':
        print(' nmf decomposition...')
        nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=200)
        stats_matrix_decomp = nmf.fit_transform(stats_sparse_matrix)

    if decomp_method == 'lda':
        print(' lda decomposition...')
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=50,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0,
                                        n_jobs=-1)
        stats_matrix_decomp = lda.fit_transform(stats_sparse_matrix)
        print(1)

    n = stats_matrix_decomp.shape[1]
    columns = ['{}_{}_{}_vector_by_{}_{}_{}'.format(decomp_method, stats_name, kinds, size, n_components, j) for j in range(n)]
    stats_df = pd.DataFrame(data=stats_matrix_decomp, columns=columns)
    print(2)
    train = stats_df[:15000].reset_index(drop=True)
    test = stats_df[15000:].reset_index(drop=True)
    for feature in columns:
        SaveFeature(train, test, feature)

    return columns, 'gen_decomposition_stats_vector_from_cat_vector("{}", "{}", "{}", "{}", {})'.format(stats_name, kinds, size, decomp_method, n_components)


if __name__ == '__main__':
    gen_decomposition_stats_vector_from_cat_vector(stats_name='mean', kinds='B', size='15d', decomp_method='lda', n_components=5)

"""
if __name__ == '__main__':

    batch_name = '20180731_am_1'
    stats_names = ['mean', 'std']
    # stats_names = ['mean', 'std']

    pool = multiprocessing.Pool(processes=3)
    func = gen_decomposition_stats_vector_ftr51
    func_name = 'gen_decomposition_stats_vector_ftr51_30d_20n'

    feature_hist_add = {}
    for result in pool.imap_unordered(func, stats_names):
        feature_names, gen_method_string = result[0], result[1]
        for feature_name in feature_names:
            feature_hist_add.setdefault(feature_name, gen_method_string)
    # 读取历史文件
    path_feature_hist = get_path() + 'FeatureGenHistory/{}.json'.format(func_name)
    feature_hist_total = json.load(open(path_feature_hist))
    # 更新字典
    feature_hist_total.setdefault(batch_name, feature_hist_add)
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)
"""