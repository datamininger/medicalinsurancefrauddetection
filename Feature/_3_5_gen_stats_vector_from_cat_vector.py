from utils import get_path, ReadData, compute_stats_dict_from_cat_matrix
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import gc


def gen_stats_vector_from_cat_vector(stats_name, size, kinds):
    """
    为了train_test_id
    :param stats_name: str, 统计名字
    :param size: str, 时间粒度
    :param kinds: str, 类别变量种类
    :return:
    """

    # 0 读取train_test_data的cat matrix
    print('gen_stats_vector_from_cat_vector("{}", "{}", "{}")'.format(stats_name, size, kinds))
    input_matrix_name = '{}_vector_from_ftr51'.format(kinds)
    input_sparse_matrix = sparse.load_npz(get_path() + 'Data/Feature/{}.npz'.format(input_matrix_name)).toarray()
    print('The shape of matrix is ( {}， {}) '.format(input_sparse_matrix.shape[0], input_sparse_matrix.shape[1]))
    # 1 读取基本数据
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
    train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 2 形成pd.dataframe， 便于分组统计
    input_sparse_df = pd.DataFrame(data=input_sparse_matrix)
    print('2')
    del input_sparse_matrix
    gc.collect()
    input_sparse_df['PERSONID'] = train_test_data['PERSONID']
    input_sparse_df['CREATETIME'] = train_test_data['CREATETIME']

    # 3 开始统计
    output_stats_df = input_sparse_df.groupby('PERSONID').apply(lambda df_person: compute_stats_dict_from_cat_matrix(df_person, stats_name, size)).to_frame(
        'stats_dict').reset_index()
    print(3)
    train_test_id = train_test_id.merge(output_stats_df, on=['PERSONID'], how='left')
    # 4 转化成稀疏矩阵并保存
    v = DictVectorizer()
    # 计算统计向量
    stats_sparse_matrix = v.fit_transform(train_test_id['stats_dict'].values)
    print(4)
    stats_matrix_name = '{}_{}_vector_by_{}'.format(stats_name, kinds, size)
    sparse.save_npz(get_path() + 'Data/Feature/{}.npz'.format(stats_matrix_name), stats_sparse_matrix)
    return stats_matrix_name, 'gen_stats_vector_from_cat_vector("{}", "{}", "{}")'.format(stats_name, size, kinds)

gen_stats_vector_from_cat_vector(stats_name='std', size='15d', kinds='B')