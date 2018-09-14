from utils import IsAbsense, SaveFeature, get_path, ReadData, compute_stats_value_FTR51_in_month, SelectDataByMonth
import multiprocessing
import json
import pandas as pd
import pdb
from sklearn.feature_extraction import DictVectorizer
from scipy import sparse


def gen_stats_value_ftr51_in_month(month='month3', stats_name='count_ratio_range'):
    """
    :param stats_name: str,对药品数量进行统计的名字
    :param size: str, 统计的时间粒度 , 7d, 15d, 30d, 45d
    :return:
    """
    # ['nunique', 'nunique_ratio', 'len', 'count_std', 'count_max', 'count_range', 'count_ratio_std', 'count_ratio_max', 'count_ratio_range']
    # pdb.set_trace()
    feature_name = '{}_ftr51_in_{}'.format(stats_name, month)
    # 0 读取数据
    train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
    train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
    train_data, test_data = SelectDataByMonth(train_data, test_data, month)
    train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    # 计算统计字典
    print('1 computing stats value of ftr51 in  {}'.format(month))
    ftr51_stats_value_df = train_test_data[['PERSONID', 'CREATETIME', 'FTR51']].groupby('PERSONID').apply(
        lambda df_person: compute_stats_value_FTR51_in_month(df_person, stats_name)).to_frame(feature_name).reset_index()
    train_test_id = train_test_id.merge(ftr51_stats_value_df, on=['PERSONID'], how='left')
    train_id[feature_name] = train_test_id[feature_name][:15000].values
    test_id[feature_name] = train_test_id[feature_name][15000:].values
    SaveFeature(train_id, test_id, feature_name)
    print('Finished Computing {} \n'.format(feature_name))
    return feature_name, 'gen_stats_value_ftr51("{}", "{}")'.format(stats_name, month)


if __name__ == '__main__':

    batch_name = '20180801_am_9'
    months = ['month{}'.format(i) for i in range(3, 15)]

    pool = multiprocessing.Pool(processes=12)
    func = gen_stats_value_ftr51_in_month
    func_name = 'gen_stats_value_ftr51_in_month'

    feature_hist_add = {}
    for result in pool.imap_unordered(func, months):
        feature_name, gen_method_string = result[0], result[1]
        feature_hist_add.setdefault(feature_name, gen_method_string)
    # 读取历史文件
    path_feature_hist = get_path() + 'FeatureGenHistory/{}.json'.format(func_name)
    feature_hist_total = json.load(open(path_feature_hist))
    # 更新字典
    feature_hist_total.setdefault(batch_name, feature_hist_add)
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)

