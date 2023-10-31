## import packages

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna

from xgboost import XGBRegressor
import lightgbm as lgb
import xgboost as xgb

from warnings import filterwarnings
filterwarnings('ignore')



def lgb_objective_ts_cv(trial, train_features, train_labels):
    """
    This function is to create LightGBM time series split validation
    """
    ## parameters
    param_grid = {
      "num_iterations": trial.suggest_int("num_iterations", 20, 100, 10),
      "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, step=0.01),
      "num_leaves": trial.suggest_int("num_leaves", 8, 72, step=1),
      "max_depth": trial.suggest_int("max_depth", 3, 8),
      "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 0.1, step=0.01),
      "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=5),
      "bagging_fraction": trial.suggest_float(
          "bagging_fraction", 0.7, 0.95, step=0.01
      ),
      "feature_fraction_bynode": trial.suggest_float(
          "feature_fraction_bynode", 0.7, 0.95, step=0.01
      ),
      "feature_fraction": trial.suggest_float(
          "bagging_fraction", 0.8, 0.95, step=0.01
      )
    }
    
    ## time series split
    tscv = TimeSeriesSplit(n_splits=5)
    folds = tscv.split(train_features)

    dtrain = lgb.Dataset(train_features, label=train_labels)

    param_grid['objective'] = "regression"
    param_grid['metric'] = "l1"
    param_grid['verbosity'] = -1
    param_grid['boosting_type'] = "gbdt"
    
    ## cross validation
    lgbcv = lgb.cv(param_grid,
                 dtrain,
                 folds=folds,
                 shuffle=False)

    cv_score = lgbcv['valid l1-mean'][-1] + lgbcv['valid l1-stdv'][-1]

    return cv_score


def lgb_model(train_features, train_labels, validation_features, validation_labels, seed, n_trials):
    """
    This function is to train the LightGBM and tune the hyperparameters
    """
    study = optuna.create_study(
            direction="minimize",
            study_name = "LightGBM Regression"
        )
    func = lambda trial: lgb_objective_ts_cv(trial, train_features, train_labels)
    study.optimize(func, n_trials = n_trials)
    lgb_parameters = study.best_params

    lgb_reg = lgb.LGBMRegressor(**lgb_parameters,
                            random_state = seed)
    lgb_reg.fit(train_features, train_labels, eval_metric = mean_absolute_error)
    Y_pred = np.expm1(lgb_reg.predict(validation_features)).astype(int).clip(0)

    print("\n\nFinal MAE for validation set is {}".format(mean_absolute_error(validation_labels, Y_pred)))
    
    return lgb_reg, lgb_parameters, study
    

def xgb_objective_ts_cv(trial, train_features, train_labels):
    """
    This function is to create XGBoost time series split validation
    """
    param_grid = {
      "n_estimators": trial.suggest_int("n_estimators", 50, 200, 10),
      "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, step=0.01),
      "max_depth": trial.suggest_int("max_depth", 3, 8),
      "subsample": trial.suggest_float("subsample", 0.7, 0.9, step=0.1),
      "min_child_weight": trial.suggest_int("min_child_weight", 5, 10, step=5),
      "colsample_bytree": trial.suggest_float(
          "colsample_bytree", 0.7, 0.95, step=0.05
      ),
      "colsample_bylevel": trial.suggest_float(
          "colsample_bylevel", 0.1, 0.3, step=0.05
      ),
      "reg_alpha": trial.suggest_float(
          "reg_alpha", 0.05, 0.2, step=0.05
      )
    }

    tscv = TimeSeriesSplit(n_splits=5)
    folds = tscv.split(train_features)
    folds_xgb = []

    for in_index, out_index in folds:
        folds_xgb.append((list(in_index), list(out_index)))

    dtrain = xgb.DMatrix(train_features, label=train_labels)

    param_grid['objective'] = "reg:squarederror"
    # param_grid['metric'] = "l1"
    param_grid['verbosity'] = 0
    param_grid['eta'] = 0.05
    param_grid['booster'] = "dart"

    xgbcv = xgb.cv(param_grid,
                 dtrain,
                 folds=folds_xgb,
                 early_stopping_rounds=50,
                 shuffle=False)

    cv_score = xgbcv['test-rmse-mean'].iloc[-1] + xgbcv['test-rmse-std'].iloc[-1]

    return cv_score
    

def xgb_model(train_features, train_labels, validation_features, validation_labels, seed, n_trials):
    """
    This function is to train the XGBoost and tune the hyperparameters
    """
    study = optuna.create_study(
        direction="minimize",
        study_name = "XGBoost"
        )

    func = lambda trial: xgb_objective_ts_cv(trial, train_features, train_labels)
    study.optimize(func, n_trials = n_trials)
    xgb_parameters = study.best_params

    xgb_reg = XGBRegressor(**xgb_parameters,
                            random_state = seed)
    xgb_reg.fit(train_features, train_labels, eval_metric = mean_absolute_error)
    Y_pred = np.expm1(xgb_reg.predict(validation_features)).astype(int).clip(0)

    print("\n\nFinal MAE for validation set is {}".format(mean_absolute_error(validation_labels, Y_pred)))

    return xgb_reg, xgb_parameters, study