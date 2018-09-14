from utils import IsAbsense, SaveFeature, get_path, ReadData, stats_cost_by_non_zero_group
import multiprocessing
import json
import pandas as pd
import pdb


def gen_stats_cost_by_non_zero_group(cost, stats_name='mean_mean', size='7d', recompute=False):
    """
    对诊疗次数的统计， 窗口可以是月或全局, 颗粒度天单位
    :param cost: str, 项目名称
    :param stats_name: str, 统计名    :param df_person:
    :param size: str, 下采样时间间隔, 类似'xd'，粒度为x天, 或 '1t'粒度为每次
    :param recompute: bool,是否重新计算该特征
    :return:
    """
    # 1
    # ['len_max', 'len_max_ratio', 'len_mean', 'len_std', 'len_count',
    #                           'sum_max', 'sum_max_ratio', 'sum_mean', 'sum_std',
    #                           'mean_max', 'mean_std', 'mean_mean']
    feature_name = '{}_{}_by_non_zero_group_{}'.format(stats_name, cost, size)
    if IsAbsense(feature_name) | recompute:
        # 2 compute feature
        print('compute {}'.format(feature_name))
        # 2.1 读取数据
        train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
        train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
        train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        # 2.3 计算count df
        stats_df = train_test_data[['PERSONID', 'CREATETIME', cost]].groupby('PERSONID').apply(
            lambda df_person: stats_cost_by_non_zero_group(df_person, cost, stats_name, size)).to_frame(feature_name).reset_index()
        # 2.4 merge 拼接
        train_test_id = train_test_id.merge(stats_df, on=['PERSONID'], how='left')
        # 2.5 保存特征
        train_id[feature_name] = train_test_id[feature_name][:15000].values
        test_id[feature_name] = train_test_id[feature_name][15000:].values
        SaveFeature(train_id, test_id, feature_name)
        print('Finished Computing {} \n'.format(feature_name))
        return feature_name, 'gen_stats_cost_by_non_zero_group("{}", "{}", "{}")'.format(cost, stats_name, size)
    else:
        print('The Feature has already been computed \n')
        return feature_name, 'gen_stats_cost_by_non_zero_group("{}", "{}", "{}")'.format(cost, stats_name, size)


if __name__ == '__main__':

    batch_name = '20180731_am_12'
    feature_matrix = ['FTR0', 'FTR2', 'FTR4', 'FTR5', 'FTR7', 'FTR8',
       'FTR9', 'FTR10', 'FTR12', 'FTR14', 'FTR16', 'FTR17', 'FTR18', 'FTR20',
       'FTR21', 'FTR23', 'FTR25', 'FTR27', 'FTR28', 'FTR29', 'FTR30', 'FTR32',
       'FTR33', 'FTR34', 'FTR35', 'FTR36', 'FTR38', 'FTR39', 'FTR40', 'FTR41',
       'FTR42', 'FTR43', 'FTR44', 'FTR45', 'FTR47', 'FTR48', 'FTR50']
    pool = multiprocessing.Pool(processes=25)
    func = gen_stats_cost_by_non_zero_group
    func_name = 'gen_stats_cost_by_non_zero_group_7d'

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
