from utils import IsAbsense, SaveFeature, get_path, ReadData, compute_stats_value_FTR51_by_size
import multiprocessing
import json
import pandas as pd
import pdb
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse


def gen_stats_value_ftr51(stats_name, size='400d'):
    """
    :param stats_name: str,对药品数量进行统计的名字
    :param size: str, 统计的时间粒度 , 7d, 15d, 30d, 45d
    :return:
    """

    feature_name = '{}_ftr51_by_{}'.format(stats_name, size)
    # 0 读取数据
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
    train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 计算统计字典
    print('1 computing stats value of ftr51 by  {}'.format(size))
    ftr51_stats_value_df = train_test_data[['PERSONID', 'CREATETIME', 'FTR51']].groupby('PERSONID').apply(
        lambda df_person: compute_stats_value_FTR51_by_size(df_person, stats_name, size)).to_frame(feature_name).reset_index()
    train_test_id = train_test_id.merge(ftr51_stats_value_df, on=['PERSONID'], how='left')
    train_id[feature_name] = train_test_id[feature_name][:15000].values
    test_id[feature_name] = train_test_id[feature_name][15000:].values
    SaveFeature(train_id, test_id, feature_name)
    print('Finished Computing {} \n'.format(feature_name))
    return feature_name, 'gen_stats_value_ftr51("{}", "{}")'.format(stats_name, size)



if __name__ == '__main__':

    batch_name = '20180801_am_5'
    stats_names = ['nunique2mean', 'nunique2max', 'nunique2min', 'nunique2std', 'nunique2range',
                  'nunique_ratio2mean', 'nunique_ratio2max', 'nunique2min', 'nunique_ratio2std', 'nunique_ratio2range',
                  'sum2mean', 'sum2max', 'sum2min', 'sum2std', 'sum2range'
                  ]

    pool = multiprocessing.Pool(processes=15)
    func = gen_stats_value_ftr51
    func_name = 'gen_stats_value_ftr51'

    feature_hist_add = {}
    for result in pool.imap_unordered(func, stats_names):
        feature_name, gen_method_string = result[0], result[1]
        feature_hist_add.setdefault(feature_name, gen_method_string)
    # 读取历史文件
    path_feature_hist = get_path() + 'FeatureGenHistory/{}.json'.format(func_name)
    feature_hist_total = json.load(open(path_feature_hist))
    # 更新字典
    feature_hist_total.setdefault(batch_name, feature_hist_add)
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)


