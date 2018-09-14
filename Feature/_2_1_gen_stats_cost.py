from utils import IsAbsense, SaveFeature, get_path, SelectDataByMonth, ReadData, cost_stats_fillna_by_stats_name, stats_cost_by_size
import multiprocessing
import json
import pandas as pd
import pdb


def gen_stats_cost(cost, stats_name='mean', month='global', size='7d', non_zero=False, recompute=True):
    """
    对诊疗次数的统计， 窗口可以是月或全局, 颗粒度天单位
    :param cost: str, 收费项目
    :param stats_name: str, 统计名
    :param size: str, 下采样时间间隔, 类似'xd'，粒度为x天, 或 '1t'粒度为每次
    :param month: str, 需要统计的时间窗口
    :param non_zero: bool, 是否为不存在的那天填充0
    :param recompute: bool,是否重新计算该特征
    :return:
    """
    # ['sum'--, 'max', 'max_ratio', 'min', 'mean', 'std', 'range']
    # ['mean', 'std', 'range', 'non_zero_count', 'non_zero_ratio', 'zero_count']
    # ['diff_mean', 'diff_max', 'diff_var', 'diff_sun']
    # 1
    feature_name = '{}_{}_in_{}_by_{}_{}'.format(stats_name, cost, month, size, non_zero)
    if IsAbsense(feature_name) | recompute:
        # 2 compute feature
        print('compute {}'.format(feature_name))
        # 2.1 读取数据
        train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
        train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
        # 2.2 选择需要统计的数据
        train_data, test_data = SelectDataByMonth(train_data, test_data, month)
        train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        # 2.3 计算count df
        stats_df = train_test_data[['PERSONID', 'CREATETIME', cost]].groupby('PERSONID').apply(
            lambda df_person: stats_cost_by_size(df_person, cost, stats_name, size, non_zero)).to_frame(feature_name).reset_index()
        # 2.4 merge 拼接
        train_test_id = train_test_id.merge(stats_df, on=['PERSONID'], how='left')
        cost_stats_fillna_by_stats_name(train_test_id, feature_name, stats_name)
        # 2.5 保存特征
        train_id[feature_name] = train_test_id[feature_name][:15000].values
        test_id[feature_name] = train_test_id[feature_name][15000:].values
        SaveFeature(train_id, test_id, feature_name)
        print('Finished Computing {} \n'.format(feature_name))
        return feature_name, 'gen_stats_cost("{}", "{}", "{}", "{}", {})'.format(cost, stats_name, month, size, non_zero)
    else:
        print('The Feature has already been computed \n')
        return feature_name, 'gen_stats_cost("{}", "{}", "{}", "{}", {})'.format(cost, stats_name, month, size, non_zero)

gen_stats_cost('FTR33')
"""
if __name__ == '__main__':

    batch_name = '20180730_am_2'
 
    feature_matrix = ['FTR0', 'FTR2', 'FTR4', 'FTR5', 'FTR7', 'FTR8',
       'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR17', 'FTR18', 'FTR20',
       'FTR21', 'FTR23', 'FTR25', 'FTR27', 'FTR28', 'FTR29', 'FTR30', 'FTR32',
       'FTR33', 'FTR34', 'FTR35', 'FTR36', 'FTR38', 'FTR39', 'FTR40', 'FTR41',
       'FTR42', 'FTR43', 'FTR44', 'FTR45', 'FTR47', 'FTR48', 'FTR50']
    pool = multiprocessing.Pool(processes=30)
    func = gen_stats_cost
    func_name = 'gen_stats_cost_diff_7d'

    feature_hist_add = {}
    for result in pool.imap_unordered(func, feature_matrix):
        feature_name, gen_method_string = result[0], result[1]
        feature_hist_add.setdefault(feature_name, gen_method_string)
    # 读取历史文件
    path_feature_hist = get_path() + 'FeatureGenHistory/{}.json'.format(func_name)
    feature_hist_total = json.load(open(path_feature_hist))
    # 更新字典
    feature_hist_total.setdefault(batch_name, feature_hist_add)
    json.dump(feature_hist_total, open(path_feature_hist, 'w'), indent=2)

"""