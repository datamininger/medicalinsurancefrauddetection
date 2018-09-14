from sklearn.ensemble import IsolationForest
from utils import CombineFeature, ReadExperimentLog, IsDifferentDistribution, SaveFeature
import pandas as pd
import pdb

def gen_isolationforest():
    pdb.set_trace()
    feature_name = 'iso_forest_score'
    # 数据准备
    log = ReadExperimentLog(43)
    config = log['config']
    Xtrain, Ytrain, Xtest = CombineFeature(config['feature_names'])
    train_test_feature = pd.concat([Xtrain, Xtest], axis=0, ignore_index=True)

    #
    clf = IsolationForest(n_estimators=500, random_state=42)
    clf.fit(train_test_feature[config['feature_names']].values)
    train_test_feature[feature_name] = clf.decision_function(train_test_feature[config['feature_names']].values)

    #
    Xtrain[feature_name] = train_test_feature[feature_name][:15000].values
    Xtest[feature_name] = train_test_feature[feature_name][15000:].values

    SaveFeature(Xtrain, Xtest, feature_name)


    IsDifferentDistribution(feature_name)

    return

gen_isolationforest()