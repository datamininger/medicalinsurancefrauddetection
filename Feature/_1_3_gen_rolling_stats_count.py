from utils import IsAbsense, SaveFeature, get_path, ReadData, rolling_stats_count
import multiprocessing
import json
import pandas as pd
import pdb


def gen_rolling_stats_count(size, stats_name='sumratio2max', recompute=False):
    """
    对诊疗次数进行滑窗统计， 窗口可以是月或全局, 颗粒度天单位
    :param stats_name: str, 统计方法
    :param size: str, 下采样时间间隔, 类似'xd'，粒度为x天,
    :param recompute: bool,是否重新计算该特征
    :return:
    """
    # 1
    feature_name = 'rolling_{}_count_{}'.format(stats_name, size)

    if IsAbsense(feature_name) | recompute:
        # 2 compute feature
        print('compute {}'.format(feature_name))
        # 2.1 读取数据
        train_id, test_id, train_data, test_data = ReadData(Ytrain=False, sort_by_time=True)
        train_test_id = pd.concat([train_id, test_id], axis=0, ignore_index=True)
        train_test_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        train_test_data['count'] = 1
        # 2.3 计算count df
        stats_df = train_test_data[['PERSONID', 'CREATETIME', 'count']].groupby('PERSONID').apply(
            lambda df_person: rolling_stats_count(df_person, stats_name, size)).to_frame(feature_name).reset_index()
        # 2.4 merge 拼接
        train_test_id = train_test_id.merge(stats_df, on=['PERSONID'], how='left')
        # 2.5 保存特征
        train_id[feature_name] = train_test_id[feature_name][:15000].values
        test_id[feature_name] = train_test_id[feature_name][15000:].values
        SaveFeature(train_id, test_id, feature_name)
        print('Finished Computing {} \n'.format(feature_name))
        return feature_name, 'gen_rolling_stats_count("{}", "{}")'.format(size, stats_name)
    else:
        print('The Feature has already been computed \n')
        return feature_name, 'gen_rolling_stats_count("{}", "{}")'.format(size, stats_name)



if __name__ == '__main__':

    batch_name = '20180731_am_2'
    feature_matrix = ['4d', '7d', '15d', '30d', '45d']
    pool = multiprocessing.Pool(processes=5)
    func = gen_rolling_stats_count
    func_name = 'gen_rolling_stats_count'

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

