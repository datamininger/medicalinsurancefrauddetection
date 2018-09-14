from utils import IsAbsense, SaveFeature, get_path, ReadData, stats_FTR51_by_size
import multiprocessing
import json
import pandas as pd
import pdb
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.externals import joblib


def gen_decomposition_stats_vector_ftr51(stats_name, size='7d', non_zero=False, decomp_method='lda', n_components=5):
    """
    :param stats_name: str,对药品数量进行统计的名字
    :param size: str, 统计的时间粒度 1d, 4d, 7d, 15d, 30d, 45d
    :param non_zero: bool, 统计是否非0
    :param decomp_method: str, 分解方法
    :param n_components: int , 分解之后的维度
    :return:
    """
    assert decomp_method in ['svd', 'nmf', 'lda']
    mask = (stats_name in ['sum', 'max', 'sum_ratio', 'max_ratio']) & non_zero
    assert not mask
    matrix_name = '{}_vector_ftr51_by_{}_{}'.format(stats_name, size, non_zero)
    # 0 读取数据

    ftr51_stats_sparse_matrix = sparse.load_npz(get_path() + 'Data/Feature/{}.npz'.format(matrix_name)).toarray()

    if decomp_method == 'svd':
        print(' svd decomposition...')
        svd = TruncatedSVD(n_components=n_components, n_iter=50, random_state=42)
        ftr51_stats_matrix_decomp = svd.fit_transform(ftr51_stats_sparse_matrix)

    if decomp_method == 'nmf':
        print(' nmf decomposition...')
        nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=200)
        ftr51_stats_matrix_decomp = nmf.fit_transform(ftr51_stats_sparse_matrix)

    if decomp_method == 'lda':
        print(' lda decomposition...')
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=50,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0,
                                        n_jobs=1)
        ftr51_stats_matrix_decomp = lda.fit_transform(ftr51_stats_sparse_matrix)
        joblib.dump(lda, "lda_{}_{}.m".format(stats_name, size))


    columns = ['{}_{}_vector_by_{}_{}_{}_{}'.format(decomp_method, stats_name, size, non_zero, n_components, j) for j in
               range(ftr51_stats_matrix_decomp.shape[1])]
    stats_df = pd.DataFrame(data=ftr51_stats_matrix_decomp, columns=columns)
    train = stats_df[:15000].reset_index(drop=True)
    test = stats_df[15000:].reset_index(drop=True)
    for feature in columns:
        SaveFeature(train, test, feature)

    return columns, 'gen_decomposition_stats_vector_ftr51("{}", "{}", {}, "{}", {})'.format(stats_name, size, non_zero, decomp_method, n_components)


gen_decomposition_stats_vector_ftr51('mean')

"""
if __name__ == '__main__':

    batch_name = '20180731_am_1'

    stats_names = ['mean', 'std',  'sum']

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
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)"""

