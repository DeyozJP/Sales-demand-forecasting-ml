from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    HistGradientBoostingRegressor, 
    ExtraTreesRegressor, 
    VotingRegressor)
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor


models = {
    'lr': LinearRegression,
    'dt': DecisionTreeRegressor,
    'rf': RandomForestRegressor,
    'hgb':HistGradientBoostingRegressor,
    'et': ExtraTreesRegressor,
    "xgb":XGBRegressor,
    'lgbm': LGBMRegressor,

   
}