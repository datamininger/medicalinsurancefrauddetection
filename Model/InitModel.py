from Model.ModelLightgbm import ModelLightgbm
from Model.ModelXgboost import ModelXgboost


def InitModel(Xtrain, Ytrain, Xtest, config):
    if config['model']['name'] == 'Xgboost':
        return ModelXgboost(Xtrain, Ytrain, Xtest, config)
    if config['model']['name'] == 'Lightgbm':
        return ModelLightgbm(Xtrain, Ytrain, Xtest, config)