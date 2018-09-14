# coding: utf-8
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import pdb
import seaborn as sns
from scipy.linalg import norm
from sklearn.feature_extraction import DictVectorizer


def get_path():
    return 'D:/xiaohuicheng_temp_i_have_no_way/medicalinsurancefrauddetection3/'


def ReadExperimentLog(log_name):
    """
    :param log_name: int , 实验记录编号
    :return: dict, log 字典
    """
    path = get_path() + 'Log/'
    with open(path + 'ExperimentLog{}.json'.format(log_name)) as f:
        log = json.load(f)
    return log


def FilterFeature(feature_names):
    """
    :param feature_names: list,
    :return: list, filtered feature names
    """
    path = get_path()

    black_list = json.load(open(path + 'BlackFeature.json'))['black_list']
    white_feature_names = []
    for feature in feature_names:
        if feature not in black_list:
            white_feature_names.append(feature)
        else:
            print(feature)
    return white_feature_names


def AddBlackFeature(feature_names):
    """
    :param feature_names: list, 新添加的黑特征
    :return:
    """
    # 2 读取
    path = get_path()
    with open(path + 'BlackFeature.json') as f:
        black_dict = json.load(f)

    black_list = black_dict['black_list']
    for feature in feature_names:
        black_list.append(feature)
    black_dict['black_list'] = black_list
    json.dump(black_dict, open(path + 'BlackFeature.json', 'w'), indent=2)


def CombineFeature(feature_names, nthreads=2):
    """
    利用特征文件名读取特征拼接成 DataFrame
    每个特征文件都是DataFrame,列名都是feature_name
    一个文件只有一个特征,feather读取单栏的速度更快
    :param feature_names: list, list of the feature file name
    :param nthreads: int
    :return: pd.DataFrame, pd.DataFrame, pd.DataFrame, Xtrain, Ytrain, Xtest
    """
    path = get_path() + "Data/Feature/"
    # 文件格式
    file_format = '.feather'
    # 空DataFrame
    Xtrain, Xtest = pd.DataFrame({}), pd.DataFrame({})
    # 1 读取训练标签
    print('Reading Ytrain...')
    Ytrain = pd.read_feather(path + 'Ytrain' + file_format)
    assert Ytrain.shape == (15000, 1)
    # 2 循环读取每个特征并赋值
    for feature_name in feature_names:
        print('     Reading {} '.format(feature_name))
        train_feature = pd.read_feather(path + feature_name + '_train' + file_format, nthreads=nthreads)
        assert train_feature.shape == (15000, 1)
        Xtrain[feature_name] = train_feature[feature_name].values

        test_feature = pd.read_feather(path + feature_name + '_test' + file_format, nthreads=nthreads)
        assert test_feature.shape == (2500, 1)
        Xtest[feature_name] = test_feature[feature_name].values

    print('Finished Combine Feature')
    return Xtrain, Ytrain, Xtest


def IsDifferentDistribution(feature, bins=100):
    """
    画分布图，人工判断是否具有不同分布，并输入数字
    :param feature:
    :return:
    """
    print('Plot the distribution of {}'.format(feature))
    Xtrain, Ytrain, Xtest = CombineFeature([feature])
    Xtrain['LABEL'] = Ytrain['LABEL'].values
    sns.distplot(Xtrain[feature].values, bins=bins, hist=True, kde=False, norm_hist=True, label='train')
    sns.distplot(Xtest[feature].values, bins=bins, hist=True, kde=False, norm_hist=True, label='test')
    plt.title(feature)
    plt.legend()
    plt.show()

    sns.violinplot(x='LABEL', y=feature, data=Xtrain)
    plt.title(feature)
    plt.show()

    IsDifferent = int(input('IsDifferent: '))
    if IsDifferent > 0:
        print('{} is a  black feature'.format(feature))
        return True
    else:
        print('{} is not  a black feature'.format(feature))
        return False


def IsAbsense(feature_name):
    """
    :param feature_name: string, 特征名字
    :return: 特征是否以及被计算,只有全都计算才返回True
    """
    path = get_path() + 'Data/Feature/'
    if os.path.exists(path + feature_name + '_train.feather') & os.path.exists(path + feature_name + '_test.feather'):
        return False
    else:
        return True


def SaveFeature(train, test, feature_name):
    """
    :param train:
    :param test:
    :param feature_name:
    :return:
    """
    path = get_path() + 'Data/Feature/'
    train[[feature_name]].to_feather(path + feature_name + '_train.feather')
    test[[feature_name]].to_feather(path + feature_name + '_test.feather')
    return


def ReadData(Ytrain=True, sort_by_time=False):
    """
    :param Ytrain: bool, 是否读取Ytrain
    :param  sort_by_time: bool, 是否按时间排序
    :return:
    """
    train_id = pd.read_feather(get_path() + '/Data/Feature/train_id.feather')
    test_id = pd.read_feather(get_path() + '/Data/Feature/test_id.feather')
    train_data = pd.read_feather(get_path() + '/Data/RawData/train.feather')
    test_data = pd.read_feather(get_path() + '/Data/RawData/test.feather')
    if sort_by_time:
        train_data = train_data.sort_values(by=['CREATETIME']).reset_index(drop=True)
        test_data = test_data.sort_values(by=['CREATETIME']).reset_index(drop=True)
    if Ytrain:
        Ytrain = pd.read_feather(get_path() + '/Data/Feature/Ytrain.feather')
        return train_id, test_id, train_data, test_data, Ytrain
    else:
        return train_id, test_id, train_data, test_data


def SelectDataByMonth(train_data, test_data, month):
    """
    :param train_data: pd.DataFrame， 提取特征的训练数据
    :param test_data:  pd.DataFrame， 提取特征的测试数据
    :param month: str, 'global'代表全局统计, 'month14'代表第35个月
    :return: train_data, test_data
    """
    assert (month == 'global') | ('month' in month)
    if month == 'global':
        return train_data, test_data
    else:
        # 1 计算月份
        train_data['month'] = train_data['CREATETIME'].dt.month + 12 * (train_data['CREATETIME'].dt.year - 2015)
        test_data['month'] = test_data['CREATETIME'].dt.month + 12 * (test_data['CREATETIME'].dt.year - 2015)
        # 2 布尔索引
        month_int = int(month[5:])
        mask_train = train_data['month'] == month_int
        mask_test = test_data['month'] == month_int
        train_data = train_data[mask_train]
        test_data = test_data[mask_test]
        train_data = train_data.drop(['month'], axis=1)
        test_data = test_data.drop(['month'], axis=1)
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        return train_data, test_data


def gen_woe(train_data, feature_list, s=0.0000000001):
    """
    :param train_data: pd.DataFrame,pd.DataFrame， 提取特征的训练数据
    :param feature_list: list, pd.DataFrame， 提取特征的测试数据
    :param s: float, 防止除0使用的
    :return: pd.DataFrame, 分组计算woe的dataframe
    """
    # pdb.set_trace()
    train_data = train_data.reset_index(drop=True)
    num_pos_total = train_data['LABEL'].sum()
    num_neg_total = train_data.shape[0] - num_pos_total
    ratio_neg_pos = (num_neg_total + s) / (float(num_pos_total) + s)

    pos_df = train_data[train_data['LABEL'] == 1].groupby(feature_list).size().to_frame('num_pos').reset_index()
    neg_df = train_data[train_data['LABEL'] == 0].groupby(feature_list).size().to_frame('num_neg').reset_index()
    woe_df = pd.merge(pos_df, neg_df, on=feature_list, how='outer')
    woe_df['num_pos'] = woe_df['num_pos'].fillna(0)
    woe_df['num_neg'] = woe_df['num_neg'].fillna(0)
    woe_df['woe'] = np.log(1 + ratio_neg_pos * ((woe_df['num_pos'] + s) / (woe_df['num_neg'] + s)))
    woe_df = woe_df.drop(['num_pos', 'num_neg'], axis=1)
    return woe_df


def gen_fraud_ratio(train_data, feature_list, s=0.00001):
    """
    :param train_data: pd.DataFrame
    :param feature_list: list,
    :param s: float
    :return:
    """
    pos_df = train_data[train_data['LABEL'] == 1].groupby(feature_list).size().to_frame('num_pos').reset_index()
    neg_df = train_data[train_data['LABEL'] == 0].groupby(feature_list).size().to_frame('num_neg').reset_index()
    fraud_ratio_df = pd.merge(pos_df, neg_df, on=feature_list, how='outer')
    fraud_ratio_df['num_pos'] = fraud_ratio_df['num_pos'].fillna(0)
    fraud_ratio_df['num_neg'] = fraud_ratio_df['num_neg'].fillna(0)
    fraud_ratio_df['fraud_ratio'] = (fraud_ratio_df['num_pos'] + s) / (fraud_ratio_df['num_neg'] + s)
    fraud_ratio_df = fraud_ratio_df.drop(['num_pos', 'num_neg'], axis=1)
    return fraud_ratio_df


def SimpleStats(df, agg_name):
    """
    :param df: pd.DataFrame， 待统计的dataframe, axis=1
    :param agg_name: 统计名字，支持 mean, var, max, min, var_diff, mean_diff, max_diff, min_diff,  var_diff_diff, mean_diff_diff, max_diff_diff, min_diff_diff
    :return: 统计值
    """
    assert agg_name in ['mean', 'var', 'max', 'min', 'var_diff', 'mean_diff', 'max_diff', 'min_diff', 'var_diff_diff',
                        'mean_diff_diff', 'max_diff_diff', 'min_diff_diff']
    # 穷举
    if agg_name == 'mean':
        stats_value = df.apply(lambda x: x.mean(), axis=1)
    if agg_name == 'var':
        stats_value = df.apply(lambda x: x.var(), axis=1)
    if agg_name == 'max':
        stats_value = df.apply(lambda x: x.max(), axis=1)
    if agg_name == 'min':
        stats_value = df.apply(lambda x: x.min(), axis=1)
    if agg_name == 'var_diff':
        stats_value = df.apply(lambda x: x.diff().var(), axis=1)
    if agg_name == 'mean_diff':
        stats_value = df.apply(lambda x: x.diff().mean(), axis=1)
    if agg_name == 'max_diff':
        stats_value = df.apply(lambda x: x.diff().max(), axis=1)
    if agg_name == 'min_diff':
        stats_value = df.apply(lambda x: x.diff().min(), axis=1)
    if agg_name == 'var_diff_diff':
        stats_value = df.apply(lambda x: x.diff().diff().var(), axis=1)
    if agg_name == 'mean_diff_diff':
        stats_value = df.apply(lambda x: x.diff().diff().mean(), axis=1)
    if agg_name == 'max_diff_diff':
        stats_value = df.apply(lambda x: x.diff().diff().max(), axis=1)
    if agg_name == 'min_diff_diff':
        stats_value = df.apply(lambda x: x.diff().diff().min(), axis=1)
    return stats_value


def compute_cost_rolling_stats(df_person, value, window_size, agg_name_inner_window, agg_name_outer_window):
    """
    :param df_person: pd.DataFrame, 按人分组以后的ddataDRAME
    :param value: str, cost 名
    :param window_size: int, 窗口大小
    :param agg_name_inner_window: str, 统计名
    :param agg_name_outer_window: str, 统计名
    :return:
    """
    assert agg_name_inner_window in ['sum', 'var', 'quantile25', 'quantile50', 'quantile75', 'mean_nonzero', 'var_nonzero', 'quantile25_nonzero',
                                     'quantile50_nonzero', 'quantile75_nonzero', 'count_nonzero', 'sum_ratio']
    assert agg_name_outer_window in ['max', 'var', 'quantile25', 'quantile50', 'quantile75']
    if agg_name_inner_window == 'sum':
        df_person = df_person.set_index('CREATETIME', drop=True)
        df_person = df_person.resample('D').sum()
        window_series = df_person[value].rolling(window=window_size, min_periods=1).sum()
    if agg_name_inner_window == 'sum_ratio':
        df_person = df_person.set_index('CREATETIME', drop=True)
        df_person = df_person.resample('D').sum()
        window_series = df_person[value].rolling(window=window_size, min_periods=1).sum()
        window_series /= df_person[value].sum()
    if agg_name_inner_window == 'mean_nonzero':
        df_person = df_person.set_index('CREATETIME', drop=True)
        df_person = df_person.resample('D').sum()
        window_series = df_person[value].rolling(window=window_size).apply(lambda x: (x[x > 0]).mean() if (x[x > 0]).shape[0] > 0 else np.nan)
    if agg_name_inner_window == 'count_nonzero':
        df_person = df_person.set_index('CREATETIME', drop=True)
        df_person = df_person.resample('D').sum()
        window_series = df_person[value].rolling(window=window_size).apply(lambda x: (x[x > 0]).shape[0])

    if agg_name_outer_window == 'max':
        status_value = window_series.max()
    if agg_name_outer_window == 'var':
        status_value = window_series.var()

    return status_value


def compute_count_rolling_stats(df_person, window_size):
    """
    :param df_person:
    :param window_size:
    :param agg_name_inner_window:
    :param agg_name_outer_window:
    :return:
    """
    df_person = df_person.set_index('CREATETIME', drop=True)
    df_person = df_person.resample('D').sum()
    window_series = df_person['count'].rolling(window=window_size, min_periods=1).sum()
    status_value = window_series.max()
    return status_value


def select_train_index_by_knn(Xtest_j, Xtrain, k):
    Xtrain['distance'] = (Xtrain - Xtest_j).apply(lambda x: norm(x.values), axis=1)
    Xtrain_sorted_k = Xtrain.sort_values(by=['distance'], axis=0)[:k]
    sampleed_train = Xtrain_sorted_k.sample(1)
    selected_index = sampleed_train.index[0]
    Xtrain = Xtrain[Xtrain.index != selected_index].drop(['distance'], axis=1)
    return selected_index, Xtrain


def gen_validate_index_by_knn(feature_names, n=5, k=8):
    """
    :param feature_names: list, list of str
    :param n: int 验证集合个数
    :param k: int 从多少个训练样本选择
    :return:
    """
    # pdb.set_trace()
    Xtrain, Ytrain, Xtest = CombineFeature(feature_names)
    mask = (Xtrain.isnull().sum() == 0) & (Xtest.isnull().sum() == 0)
    columns = list(Xtrain.columns[mask])
    Xtrain = Xtrain[columns]
    Xtest = Xtest[columns]
    # min-max 归一化
    for feature in columns:
        max_value, min_value = max(Xtrain[feature].max(), Xtest[feature].max()), min(Xtrain[feature].min(), Xtest[feature].min())
        Xtrain[feature] = (Xtrain[feature] - min_value) / float((max_value - min_value))
        Xtest[feature] = (Xtest[feature] - min_value) / float((max_value - min_value))

    #
    index_matrix = []
    for i in range(n):
        index_list = []
        Xtrain_1 = Xtrain.copy(deep=True)
        Xtest_1 = Xtest.copy(deep=True)

        for j in range(Xtest.shape[0]):
            index, Xtrain_1 = select_train_index_by_knn(Xtest_1.iloc[j], Xtrain_1, k)
            index_list.append(index)
        index_matrix.append(index_list)
    return index_matrix


def read_valid_index(name):
    """
    :param name: int
    :return: list, list, train_index_matrix, valid_index_matrix
    """
    valid_index_df = pd.read_feather(get_path() + 'validate_index/' + 'validate_index_{}.feather'.format(name))
    valid_index_matrix = [list(valid_index_df[str(i)].values) for i in range(1, 6)]
    train_index_matrix = []
    for valid_index_list in valid_index_matrix:
        train_index_list = list(set(range(15000)) - set(valid_index_list))
        train_index_list.sort()
        train_index_matrix.append(train_index_list)
    return train_index_matrix, valid_index_matrix


def compute_fold_result_xgboost(booster_offline, evals_result, feature_names, norm):
    """
    :param booster_offline: booster,
    :param evals_result:dict,
    :param feature_names:list,
    :return: dict
    """
    best_iteration = booster_offline.best_iteration
    score_offline = evals_result['valid']['auc'][best_iteration]
    fscore_dict = booster_offline.get_fscore()
    if norm:
        feat_sum_imp = np.sum([fscore_dict.setdefault(feature, 0) for feature in feature_names])
        feature_importance_dict = {feature: (fscore_dict.setdefault(feature, 0) / float(feat_sum_imp)) for feature in feature_names}
    else:
        feature_importance_dict = {feature: fscore_dict.setdefault(feature, 0) for feature in feature_names}

    fold_result = {'best_iteration': best_iteration,
                   'score_offline': score_offline,
                   'feature_importance_dict': feature_importance_dict}
    return fold_result


def compute_fold_result_lightgbm(booster_offline, evals_result, feature_names, norm):
    """
    :param booster_offline:
    :param evals_result:
    :param feature_names:
    :param norm:bool
    :return:
    """
    best_iteration = booster_offline.best_iteration
    score_offline = evals_result['valid']['auc'][best_iteration]
    feature_importance_list = booster_offline.feature_importance(iteration=booster_offline.best_iteration)
    if norm:
        feature_importance_list = np.array(feature_importance_list) / np.sum(feature_importance_list)
    feature_importance_dict = {feature: float(importance) for feature, importance in zip(feature_names, feature_importance_list)}
    fold_result = {'best_iteration': best_iteration,
                   'score_offline': score_offline,
                   'feature_importance_dict': feature_importance_dict}
    return fold_result


def ensemble_submission(submissions, weights, ensemble_method):
    """
    :param submissions: list,
    :param weights: list
    :param ensemble_method: str
    :return: list
    """
    assert ensemble_method in ['mean', 'rank_mean', 'mean_weight', 'rank_mean_weight']
    print('ensemble method is {}'.format(ensemble_method))
    if ensemble_method == 'mean':
        submission_online = np.mean(submissions, axis=0)
    if ensemble_method == 'rank_mean':
        submission_set = []
        for submission in submissions:
            submission_set.append(pd.Series(submission).rank().values)
        submission_online = np.mean(submission_set, axis=0)
    if ensemble_method == 'mean_weight':
        submission_set = []
        for submission, weight in zip(submissions, weights):
            submission_set.append(weight * pd.Series(submission).values)
        submission_online = np.sum(submission_set, axis=0) / np.sum(weights)
    if ensemble_method == 'rank_mean_weight':
        submission_set = []
        for submission, weight in zip(submissions, weights):
            submission_set.append(weight * pd.Series(submission).rank().values)
        submission_online = np.sum(submission_set, axis=0) / np.sum(weights)
    return submission_online


def stats_count_by_size(df_person, stats_name, size, non_zero):
    """
    按颗粒度求和、过滤、统计
    :param df_person: pd.Dataframe, 按PERSONID分组以后的片段
    :param stats_name: str, 统计名字
    :param size: str, 粒度
    :param non_zero: bool, 统计操作只包括非0的颗粒度
    :return: float, 统计值
    """
    # 0
    assert stats_name in ['sum', 'max', 'max_ratio', 'min', 'mean', 'std', 'range']
    assert 'd' in size
    mask1 = ((stats_name in ['sum', 'max', 'max_ratio', 'min']) & non_zero)
    assert not mask1
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    df_person_resampled = df_person_resampled.resample(size).sum()

    # 2 是否过滤值为0的颗粒度
    if non_zero:
        df_person_resampled = df_person_resampled[df_person_resampled['count'] > 0]
    else:
        pass

    # 3 使用规则，开始统计,注意缺失值, 颗粒度至少有一个，不用处理空dataframe
    if stats_name == 'sum':
        # 没有缺失值的
        stats_value = df_person_resampled['count'].sum()

    elif stats_name == 'max':
        stats_value = df_person_resampled['count'].max()

    elif stats_name == 'max_ratio':
        stats_value = (df_person_resampled['count'].max() / df_person_resampled['count'].sum())

    elif stats_name == 'min':
        stats_value = df_person_resampled['count'].min()

    elif stats_name == 'mean':
        stats_value = df_person_resampled['count'].mean()

    elif stats_name == 'std':
        # 当只有一个颗粒度的时候，返回缺失值,标准差肯定大于0，使用-1.5是不错的选择
        stats_value = df_person_resampled['count'].std()
        if pd.isnull(stats_value):
            stats_value = -1

    elif stats_name == 'range':
        stats_value = df_person_resampled['count'].max() - df_person_resampled['count'].min()

    return stats_value


def count_stats_fillna_by_stats_name(train_test_id, feature_name, stats_name):
    """
    此时的填充意味在这个月份窗口，该人员没有出现
    :param train_test_id: pd.DataFrame,
    :param  feature_name: str,
    :param stats_name: str,
    :return:
    """
    assert stats_name in ['sum', 'max', 'max_ratio', 'min', 'mean', 'std', 'range']
    if stats_name == 'sum':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(0)
    elif stats_name == 'max':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(0)
    elif stats_name == 'max_ratio':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(0)
    elif stats_name == 'min':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(0)
    elif stats_name == 'mean':
        # 树模型也许便于划分
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)
    elif stats_name == 'std':
        # 树模型也许便于划分，在统计时填充的是-1 进一步区分填充-2
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-2)
    elif stats_name == 'range':
        # 树模型也许便于划分
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)


def stats_cost_by_size(df_person, cost, stats_name, size, non_zero):
    """
    :param df_person:
    :param cost:
    :param stats_name:
    :param size:
    :param non_zero:
    :return:
    """
    # 0
    #  只支持以下统计
    assert stats_name in ['sum', 'max', 'max_ratio', 'min', 'mean', 'std', 'range', 'non_zero_count', 'non_zero_ratio', 'zero_count',
                          'diff_mean', 'diff_max', 'diff_sum', 'diff_var']
    # 以下统计部考虑非0
    mask1 = ((stats_name in ['sum', 'max', 'max_ratio', 'min']) & non_zero)
    assert not mask1
    # 窗口只有天或次
    assert ('d' in size) | ('1t' == size)
    # 使用以下统计时，必须开启非0
    if stats_name in ['non_zero_count', 'non_zero_ratio', 'zero_count']:
        assert non_zero
    df_person = df_person.reset_index(drop=True)
    m = df_person.shape[0]
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数
    if size != '1t':
        # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
        df_person_resampled = df_person.resample('1d').sum()
        print(df_person_resampled)
        df_person_resampled = df_person_resampled.resample(size).sum()
        print(df_person_resampled)
        pdb.set_trace()
    else:
        df_person_resampled = df_person.copy()

    # 2 是否过滤值为0的颗粒度
    if non_zero:
        df_person_resampled = df_person_resampled[df_person_resampled[cost] > 0]
    else:
        pass

    # 3 使用规则，开始统计,注意缺失值, 颗粒度至少有一个，不用处理空dataframe
    if stats_name == 'sum':
        # 没有缺失值的
        stats_value = df_person_resampled[cost].sum()

    elif stats_name == 'max':
        # 没有缺失值的
        stats_value = df_person_resampled[cost].max()

    elif stats_name == 'max_ratio':
        if df_person_resampled[cost].sum() == 0:
            # 全是0，没有算比例的必要
            stats_value = -1
        else:
            stats_value = (df_person_resampled[cost].max() / df_person_resampled[cost].sum())

    elif stats_name == 'min':
        stats_value = df_person_resampled[cost].min()
    # 以下可统计非0
    elif stats_name == 'mean':
        if df_person_resampled.shape[0] == 0:
            stats_value = -1
        else:
            stats_value = df_person_resampled[cost].mean()

    elif stats_name == 'std':
        if df_person_resampled.shape[0] == 0:
            stats_value = -1
        else:
            # 当只有一个颗粒度的时候，返回缺失值,标准差肯定大于0，使用-1.5是不错的选择
            stats_value = df_person_resampled[cost].std()
            if pd.isnull(stats_value):
                stats_value = -1.5

    elif stats_name == 'range':
        if df_person_resampled.shape[0] == 0:
            stats_value = -1
        else:
            stats_value = df_person_resampled[cost].max() - df_person_resampled[cost].min()

    elif stats_name == 'non_zero_count':
        if df_person_resampled.shape[0] == 0:
            stats_value = 0
        else:
            stats_value = df_person_resampled.shape[0]

    elif stats_name == 'non_zero_ratio':
        if df_person_resampled.shape[0] == 0:
            stats_value = 0.0
        else:
            stats_value = df_person_resampled.shape[0] / float(m)

    elif stats_name == 'zero_count':
        if df_person_resampled.shape[0] == 0:
            stats_value = m
        else:
            stats_value = m - df_person_resampled.shape[0]

    elif stats_name == 'diff_mean':
        if df_person_resampled.shape[0] == 0:
            stats_value = -100
        else:
            stats_value = df_person_resampled[cost].diff().mean()
            if pd.isnull(stats_value):
                stats_value = -150

    elif stats_name == 'diff_sum':
        if df_person_resampled.shape[0] == 0:
            stats_value = -100
        else:
            stats_value = df_person_resampled[cost].diff().sum()
            if pd.isnull(stats_value):
                stats_value = -150

    elif stats_name == 'diff_var':
        if df_person_resampled.shape[0] == 0:
            stats_value = -1
        else:
            stats_value = df_person_resampled[cost].diff().sum()
            if pd.isnull(stats_value):
                stats_value = -1.5

    elif stats_name == 'diff_max':
        if df_person_resampled.shape[0] == 0:
            stats_value = -10
        else:
            stats_value = df_person_resampled[cost].diff().max()
            if pd.isnull(stats_value):
                stats_value = -15
    return stats_value


def cost_stats_fillna_by_stats_name(train_test_id, feature_name, stats_name):
    """
    此时的填充意味在这个月份窗口，该人员没有出现
    :param train_test_id: pd.DataFrame,
    :param  feature_name: str,
    :param stats_name: str,
    :return:
    """
    assert stats_name in ['sum', 'max', 'max_ratio', 'min', 'mean', 'std', 'range', 'non_zero_count', 'non_zero_ratio', 'zero_count',
                          'diff_mean', 'diff_max', 'diff_sum', 'diff_var']
    if stats_name == 'sum':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'max':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'max_ratio':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'min':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'mean':
        # 树模型也许便于划分-2
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-2)

    elif stats_name == 'std':
        # 树模型也许便于划分，在统计时填充的是-1 进一步区分填充-2
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-2)

    elif stats_name == 'range':
        # 树模型也许便于划分
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-2)

    elif stats_name == 'non_zero_count':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'non_zero_ratio':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)

    elif stats_name == 'zero_count':
        train_test_id[feature_name] = train_test_id[feature_name].fillna(-1)
    else:
        pass
    return


def compute_non_zero_group(arr):
    """
    [1,0,0,3,4,0] - [[1],[3,4]]
    [0,4,6,0,0,1] - [[4,6],[1]]
    [0,4,6,0,1] - [[4,6],[1]]
    [1,2,3]     - [[1,2,3]]
    [0, 0, 0]  - []
    [0,0,0,1]   [[1]]
    [1,0,0]   [[1]]
    ['a', 0, 'bjhkjhj']
    非0组用0隔开
    :param arr: list, 元素为字符串或字符
    :return: list, 非0组集合
    """
    # 0 补零
    arr_1 = arr[:]
    arr_1.append(0)
    group_set = []
    start = np.nan
    for i, number in enumerate(arr_1):
        if pd.isna(start):  # 是否有起点
            if number != 0:  # 目前是否合适
                start = i
            else:
                continue  # 下一个，直到有起点
        else:
            if number == 0:  # 如果是0，则为终点
                group_set.append(arr_1[start:i])
                start = np.nan
            else:
                continue  # 下一个，直到end
    return group_set


def stats_count_by_non_zero_group(df_person, stats_name, size):
    """
    :param df_person: pd.Dataframe
    :param stats_name: str,
    :param size: str
    :return:
    """
    assert stats_name in ['len_max', 'len_max_ratio', 'len_mean', 'len_std', 'len_count',
                          'sum_max', 'sum_max_ratio', 'sum_mean', 'sum_std',
                          'mean_max', 'mean_std', 'mean_mean']
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    df_person_resampled = df_person_resampled.resample(size).sum()
    print(df_person_resampled)
    # 2 计算non zero group
    non_zero_group = compute_non_zero_group(df_person_resampled['count'].tolist())

    # 3 开始统计，利用规则
    # ------- 长度向量统计
    if stats_name == 'len_max':
        len_vector = pd.Series([len(group) for group in non_zero_group])
        stats_value = len_vector.max()

    elif stats_name == 'len_max_ratio':
        len_vector = pd.Series([len(group) for group in non_zero_group])
        stats_value = len_vector.max() / len_vector.sum()

    elif stats_name == 'len_mean':
        len_vector = pd.Series([len(group) for group in non_zero_group])
        stats_value = len_vector.mean()

    elif stats_name == 'len_std':
        len_vector = pd.Series([len(group) for group in non_zero_group])
        stats_value = len_vector.std()
        if pd.isnull(stats_value):
            stats_value = -1

    elif stats_name == 'len_count':
        len_vector = pd.Series([len(group) for group in non_zero_group])
        stats_value = len_vector.count()

    # ------- 和向量统计

    elif stats_name == 'sum_max':
        sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
        stats_value = sum_vector.max()

    elif stats_name == 'sum_max_ratio':
        sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
        stats_value = sum_vector.max() / sum_vector.sum()

    elif stats_name == 'sum_mean':
        sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
        stats_value = sum_vector.mean()

    elif stats_name == 'sum_std':
        sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
        stats_value = sum_vector.std()
        if pd.isnull(stats_value):
            stats_value = -1

    # ------ 平均向量统计
    elif stats_name == 'mean_max':
        mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
        stats_value = mean_vector.max()

    elif stats_name == 'mean_std':
        mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
        stats_value = mean_vector.std()
        if pd.isnull(stats_value):
            stats_value = -1

    elif stats_name == 'mean_mean':
        mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
        stats_value = mean_vector.mean()

    assert not pd.isnull(stats_value)
    return stats_value


def stats_cost_by_non_zero_group(df_person, cost, stats_name, size):
    """
    :param df_person:
    :param cost:
    :param stats_name:
    :param size:
    :return:
    """
    assert stats_name in ['len_max', 'len_max_ratio', 'len_mean', 'len_std', 'len_count',
                          'sum_max', 'sum_max_ratio', 'sum_mean', 'sum_std',
                          'mean_max', 'mean_std', 'mean_mean']
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    df_person_resampled = df_person_resampled.resample(size).sum()
    # 2 计算non zero group
    non_zero_group = compute_non_zero_group(df_person_resampled[cost].tolist())

    # 3 开始统计，利用规则
    # ------- 长度向量统计
    if stats_name == 'len_max':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            len_vector = pd.Series([len(group) for group in non_zero_group])
            stats_value = len_vector.max()

    elif stats_name == 'len_max_ratio':
        if len(non_zero_group) == 0:
            stats_value = -1
        else:
            len_vector = pd.Series([len(group) for group in non_zero_group])
            stats_value = len_vector.max() / len_vector.sum()

    elif stats_name == 'len_mean':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            len_vector = pd.Series([len(group) for group in non_zero_group])
            stats_value = len_vector.mean()

    elif stats_name == 'len_std':
        if len(non_zero_group) == 0:
            stats_value = -1
        else:
            len_vector = pd.Series([len(group) for group in non_zero_group])
            stats_value = len_vector.std()
            if pd.isnull(stats_value):
                stats_value = -1.5

    elif stats_name == 'len_count':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            len_vector = pd.Series([len(group) for group in non_zero_group])
            stats_value = len_vector.count()

    # ------- 和向量统计

    elif stats_name == 'sum_max':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
            stats_value = sum_vector.max()

    elif stats_name == 'sum_max_ratio':
        if len(non_zero_group) == 0:
            stats_value = -1
        else:
            sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
            stats_value = sum_vector.max() / sum_vector.sum()

    elif stats_name == 'sum_mean':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
            stats_value = sum_vector.mean()

    elif stats_name == 'sum_std':
        if len(non_zero_group) == 0:
            stats_value = -1
        else:
            sum_vector = pd.Series([np.sum(group) for group in non_zero_group])
            stats_value = sum_vector.std()
            if pd.isnull(stats_value):
                stats_value = -1.5

    # ------ 平均向量统计
    elif stats_name == 'mean_max':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
            stats_value = mean_vector.max()

    elif stats_name == 'mean_std':
        if len(non_zero_group) == 0:
            stats_value = -1
        else:
            mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
            stats_value = mean_vector.std()
            if pd.isnull(stats_value):
                stats_value = -1.5

    elif stats_name == 'mean_mean':
        if len(non_zero_group) == 0:
            stats_value = 0
        else:
            mean_vector = pd.Series([np.mean(group) for group in non_zero_group])
            stats_value = mean_vector.mean()

    assert not pd.isnull(stats_value)
    return stats_value


def rolling_stats_count(df_person, stats_name, size):
    """
    :param df_person:
    :param stats_name:
    :param size:
    :return:
    """
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    if stats_name == 'sum2max':
        rolling_df = df_person_resampled.rolling(size).sum()
        stats_value = rolling_df['count'].max()
    if stats_name == 'sumratio2max':
        rolling_df = df_person_resampled.rolling(size).sum()
        stats_value = rolling_df['count'].max()
        stats_value /= (df_person['count'].sum() + 0.000000001)
    assert not pd.isnull(stats_value)
    return stats_value


def rolling_stats_cost(df_person, cost, stats_name, size):
    """
    :param df_person:
    :param cost:
    :param stats_name:
    :param size:
    :return:
    """
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    if stats_name == 'sum2max':
        rolling_df = df_person_resampled.rolling(size).sum()
        stats_value = rolling_df[cost].max()
    if stats_name == 'sumratio2max':
        rolling_df = df_person_resampled.rolling(size).sum()
        stats_value = rolling_df[cost].max()
        stats_value /= (df_person[cost].sum() + 0.000000001)

    assert not pd.isnull(stats_value)
    return stats_value


def ftr51_mean(s, non_zero):
    """
    :param s: pd.series
    :param non_zero: bool
    :return:
    """
    if not non_zero:
        return s.mean()
    else:
        mask = (s != 0)
        return s[mask].mean()


def ftr51_std(s, non_zero):
    """
    :param s: pd.series
    :param non_zero: bool
    :return:
    """
    if not non_zero:
        stats_value = s.std()
    else:
        mask = (s != 0)
        if np.sum(mask) > 0:
            stats_value = s[mask].std()
        else:
            stats_value = 0
    if pd.isna(stats_value):
        stats_value = 0

    return stats_value


def stats_FTR51_by_size(df_person, stats_name, size, non_zero):
    """
    :param df_person:
    :param stats_name:
    :param size:
    :param non_zero:
    :return:
    """

    assert stats_name in ['sum', 'sum_ratio', 'max', 'max_ratio', 'mean', 'std']
    mask = (stats_name in ['sum', 'sum_ratio', 'max', 'max_ratio']) & non_zero
    assert not mask

    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    #print(df_person)
    # 0 形成矩阵
    df_person['FTR51_count_dict'] = df_person['FTR51'].map(lambda char: pd.value_counts(char.split(',')).to_dict())
    v = DictVectorizer()
    ftr51_count_matrix = v.fit_transform(df_person['FTR51_count_dict'].values).toarray()

    columns = v.get_feature_names()
    ftr51_count_df = pd.DataFrame(data=ftr51_count_matrix, columns=columns, index=df_person.index)
    # 1 形成颗粒
    ftr51_count_df_resampled = ftr51_count_df.resample('1d').sum()
    ftr51_count_df_resampled = ftr51_count_df_resampled.resample(size).sum()


    # 2 准备统计
    if stats_name == 'sum':
        stats_dict = ftr51_count_df_resampled.sum(axis=0).to_dict()

    elif stats_name == 'sum_ratio':
        stats_series = ftr51_count_df_resampled.sum(axis=0)
        stats_series /= stats_series.sum()
        stats_dict = stats_series.to_dict()

    elif stats_name == 'max':
        stats_dict = ftr51_count_df_resampled.max(axis=0).to_dict()

    elif stats_name == 'max_ratio':
        stats_series = ftr51_count_df_resampled.max(axis=0)
        stats_series /= ftr51_count_df_resampled.sum(axis=0)
        stats_dict = stats_series.to_dict()

    elif stats_name == 'mean':
        stats_series = ftr51_count_df_resampled.apply(lambda s: ftr51_mean(s, non_zero), axis=0)
        stats_dict = stats_series.to_dict()

    elif stats_name == 'std':
        stats_series = ftr51_count_df_resampled.apply(lambda s: ftr51_std(s, non_zero), axis=0)
        stats_dict = stats_series.to_dict()

    return stats_dict


def take_cat(ftr51, kinds):
    """
    take_cat('A24B176C1239E0D0', 'BD') -> B176_D0
    take_cat('A24B176C1239E0D0', 'E') -> E0

    :param ftr51: str
    :param kinds: str
    :return:
    """
    # 补一个方便索引
    ftr51 += '_'
    took_cat_list = []
    # pdb.set_trace()
    for kind in kinds:
        assert kind in ['A', 'B', 'C', 'D', 'E']
        # 0 下标初始化
        start_index = np.nan
        end_index = np.nan
        # 1 计算下标
        start_index = ftr51.find(kind)
        if kind == 'A':
            end_index = ftr51.find('B')
        elif kind == 'B':
            end_index = ftr51.find('C')
        elif kind == 'C':
            end_index = ftr51.find('E')
        elif kind == 'D':
            end_index = -1
        elif kind == 'E':
            end_index = ftr51.find('D')
        # 2 索引提取
        took_cat_list.append(ftr51[start_index: end_index])
    took_cat = '_'.join(took_cat_list)
    return took_cat


def exam_kinds_sort(kinds):
    """
    检查特征组合是否已排序
    :param kinds:
    :return:
    """
    kinds_list = list(kinds)
    kinds_list.sort()
    assert ''.join(kinds_list) == kinds


def compute_cat_count_dict_from_ftr51s(ftr51s, kinds):
    """
    将某种类别变量从FTR51s中提取出来, 计数成字典
    'A0B0C428E0D0,A0B0C82E0D0,A0B0C465E0D0,A0B0C95E0D0' kind='AC' -> ['A0_C428', 'A0_C82', 'A0_C465', 'A0_C95']
    :param ftr51s: str, ftr51特征
    :param kinds: str, 需要提取的cat, A, B, AB等, AB代表组合类别变量
    :return:
    """
    exam_kinds_sort(kinds)
    cat_list = []
    ftr51_list = ftr51s.split(',')
    for ftr51 in ftr51_list:
        took_cat = take_cat(ftr51, kinds)
        cat_list.append(took_cat)
    return pd.value_counts(cat_list).to_dict()


def compute_stats_dict_from_cat_matrix(df_person, stats_name, size):
    """
    :param df_person:
    :param stats_name:
    :param size:
    :return:
    """
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 1 计算颗粒度内的诊疗次数

    # 计算颗粒度的就诊次数和; 没有出现的天数为0， 再降采样,
    df_person_resampled = df_person.resample('1d').sum()
    df_person_resampled = df_person_resampled.resample(size).sum()

    if stats_name == "sum":
        stats_series = df_person_resampled.sum(axis=0)

    elif stats_name == 'max':
        stats_series = df_person_resampled.max(axis=0)

    elif stats_name == 'std':
        stats_series = df_person_resampled.std(axis=0)

    elif stats_name == 'mean':
        stats_series = df_person_resampled.mean(axis=0)
    stats_series = stats_series[stats_series != 0]
    return stats_series.to_dict()


def compute_stats_value_FTR51_by_size(df_person, stats_name, size):
    """
    :param df_person: pd.DataFrame
    :param stats_name: str,
    :param size: str
    :return:
    """
    assert stats_name in ['nunique2mean', 'nunique2max', 'nunique2min', 'nunique2std', 'nunique2range',
                          'nunique_ratio2mean', 'nunique_ratio2max', 'nunique2min', 'nunique_ratio2std', 'nunique_ratio2range',
                          'sum2mean', 'sum2max', 'sum2min', 'sum2std', 'sum2range'
                          ]
    df_person = df_person.reset_index(drop=True)
    df_person = df_person.set_index('CREATETIME', drop=True)
    # 0 形成矩阵
    df_person['FTR51_count_dict'] = df_person['FTR51'].map(lambda char: pd.value_counts(char.split(',')).to_dict())
    v = DictVectorizer()
    ftr51_count_matrix = v.fit_transform(df_person['FTR51_count_dict'].values).toarray()
    columns = v.get_feature_names()
    ftr51_count_df = pd.DataFrame(data=ftr51_count_matrix, columns=columns, index=df_person.index)
    # 1 形成颗粒
    ftr51_count_df_resampled = ftr51_count_df.resample('1d').sum()
    ftr51_count_df_resampled = ftr51_count_df_resampled.resample(size).sum()
    # 2 开始统计
    stats_name_1, stats_name_2 = stats_name.split('2')[0], stats_name.split('2')[1]

    # 药品种类个数
    if stats_name_1 == 'nunique':
        stats_series = ftr51_count_df_resampled.apply(lambda s: (s != 0).sum(), axis=1)
    # 药品集中度
    elif stats_name_1 == 'nunique_ratio':
        stats_series = ftr51_count_df_resampled.apply(lambda s: (s != 0).sum(), axis=1)
        sum_total = ftr51_count_df_resampled.sum(axis=1) + 0.000001
        stats_series /= sum_total
    # 药品总盒数
    elif stats_name_1 == 'sum':
        stats_series = ftr51_count_df_resampled.sum(axis=1)
    # 平均
    if stats_name_2 == 'mean':
        stats_value = stats_series.mean()
    # 极值
    elif stats_name_2 == 'max':
        stats_value = stats_series.max()
    # 极值
    elif stats_name_2 == 'min':
        stats_value = stats_series.min()
    # 波动
    elif stats_name_2 == 'std':
        stats_value = stats_series.std()
        if pd.isna(stats_value):
            stats_value = -1
    # 波动范围
    elif stats_name_2 == 'range':
        stats_value = stats_series.max() - stats_series.min()

    return stats_value


def compute_stats_value_FTR51_in_month(df_person, stats_name):
    """
    :param df_person:
    :param stats_name:
    :return:
    """
    assert stats_name in ['nunique', 'nunique_ratio', 'len', 'count_std', 'count_max', 'count_range', 'count_ratio_std', 'count_ratio_max', 'count_ratio_range']

    if stats_name == 'nunique_ratio':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = len(set(ftr51_list))/float(len(ftr51_list))

    elif stats_name == 'nunique':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = len(set(ftr51_list))

    elif stats_name == 'len':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = len(ftr51_list)

    elif stats_name == 'count_std':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = pd.value_counts(ftr51_list).std()
        if pd.isna(stats_value):
            stats_value = -1

    elif stats_name == 'count_max':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = pd.value_counts(ftr51_list).max()

    elif stats_name == 'count_range':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        stats_value = pd.value_counts(ftr51_list).max() - pd.value_counts(ftr51_list).min()

    elif stats_name == 'count_ratio_std':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        count_ratio = pd.value_counts(ftr51_list)/pd.value_counts(ftr51_list).sum()
        stats_value = count_ratio.std()
        if pd.isna(stats_value):
            stats_value = -1

    elif stats_name == 'count_ratio_max':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        max_ratio = pd.value_counts(ftr51_list).max()/pd.value_counts(ftr51_list).sum()
        stats_value = max_ratio.max()

    elif stats_name == 'count_ratio_range':
        ftr51_list = ','.join(list(df_person['FTR51'].values))
        count_ratio = pd.value_counts(ftr51_list)/pd.value_counts(ftr51_list).sum()
        stats_value = count_ratio.max() - count_ratio.min()

    assert  not pd.isna(stats_value)

    return stats_value


def stats_by_oob_dict(s, stats_name):
    """
    :param s:
    :param stats_name:
    :return:
    """

    if stats_name == 'fraud_count':
        stats_value = pd.Series(s['count_dict_oob']).sum()

    if stats_name == 'fraud_ratio_sum':
        fraud_series_oob = pd.Series(s['fraud_dict_oob'])
        count_series_oob = pd.Series(s['count_dict_oob'])

        assert (fraud_series_oob.isnull().sum() == 0)
        assert (count_series_oob.isnull().sum() == 0)

        fraud_ratio_series_oob = fraud_series_oob.div(count_series_oob + 0.000000001)
        stats_value = fraud_ratio_series_oob.sum()
    if stats_name == 'fraud_ratio_mean_weight':
        fraud_series_oob = pd.Series(s['fraud_dict_oob'])
        count_series_oob = pd.Series(s['count_dict_oob'])

        assert (fraud_series_oob.isnull().sum() == 0)
        assert (count_series_oob.isnull().sum() == 0)

        fraud_ratio_series_oob = fraud_series_oob.div(count_series_oob + 0.000000001)
        count_series_person = pd.Series(s['count_dict_person'])
        stats_value = (fraud_ratio_series_oob.mul(count_series_person)).sum()/(count_series_person.sum()+0.000000001)


    return stats_value



