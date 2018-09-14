from utils import IsAbsense, SaveFeature, get_path, SelectDataByMonth, ReadData
import multiprocessing
import json

def gen_(month, recompute=False):
    # 1
    feature_name = ''


    if IsAbsense(feature_name) | recompute:
        # 2 compute feature
        print('compute {}'.format(feature_name))
        # 2.1 读取数据
        train_id, test_id, train_data, test_data, Ytrain = ReadData(Ytrain=True)
        train_id['LABEL'] = Ytrain['LABEL'].values
        train_data = train_data.merge(train_id, on=['PERSONID'], how='left')
        # 2.2 选择需要统计的数据
        train_data, test_data = SelectDataByMonth(train_data, test_data, month)
        # 如果本月未出现
        train_id[feature_name] = train_id[feature_name].fillna(0)
        test_id[feature_name] = test_id[feature_name].fillna(0)
        # 保存特征
        SaveFeature(train_id, test_id, feature_name)
        print('Finished Computing {} \n'.format(feature_name))
        return feature_name, 'gen_stats_woe_OfPerson_by_columns({}, "{}", {})'.format(feature_list, agg_name, False)
    else:
        print('The Feature has already been computed \n')
        return feature_name, 'gen_stats_woe_OfPerson_by_columns({}, "{}", {})'.format(feature_list, agg_name, False)


if __name__ == '__main__':

    batch_name = '201807276987'
    feature_matrix = [['FTR5', 'FTR6']]
    pool = multiprocessing.Pool(processes=6)
    func = gen_stats_woe_columns_OfPerson
    func_name = 'gen_stats_woe_columns_OfPerson'

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








