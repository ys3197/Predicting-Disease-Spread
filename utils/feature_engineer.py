## import packages
# %matplotlib inline

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')


## weekly median total counts
def create_weekly_features(dataset):
  """
  This function is to creat weekly features
  """
  updated_df = pd.DataFrame()

  for year in dataset.year.unique():
    data_cur_year = dataset[dataset.year == year]
    ## no historical data for first year
    if year == dataset.year.unique()[0]:
      data_cur_year['weekly_median_cases'] = None
      # data_cur_year['monthly_ave_cases'] = None

    ## only focusing on historical data to avoid data leakage
    else:
      week_median = dataset[dataset.year < year].groupby(['weekofyear'])['total_cases'].median().to_frame().rename(columns={'total_cases': 'weekly_median_cases'}).reset_index()
      # month_ave = dataset[dataset.year < year].groupby(['month'])['total_cases'].mean().to_frame().rename(columns={'total_cases': 'monthly_ave_cases'}).reset_index()
      data_cur_year = data_cur_year.merge(week_median, how='left', on='weekofyear')
      # data_cur_year = data_cur_year.merge(month_ave, how='left', on='month')

    updated_df = pd.concat([updated_df, data_cur_year], ignore_index=True)

  return updated_df


def remove_features(dataset):
  """
  This function is to remove some unnecessary features
  """
  ## drop tempearture related features
  dataset.drop(columns = ['reanalysis_avg_temp_k',
                          # 'reanalysis_tdtr_k',
                          'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg'], inplace=True)
  ## drop columns with > 20 % missing values
  dataset.drop(columns = ['ndvi_ne'], inplace=True)
  ## drop some time-related features
  dataset.drop(columns = ['weekofyear', 'week_start_date', 'year'], inplace=True)

  return dataset


def feature_crossing(dataset):
  """
  This function is to create some feature interactions
  """
  dataset['reanalysis_hot_humid_index'] = dataset['reanalysis_relative_humidity_percent']*dataset['reanalysis_tdtr_k']

  return dataset


def other_preprocess(dataset, multiple_models=False):
  """
  This function is to fill missing values and encode city column
  """
  ## Fill the rest using linear interpolation
  dataset.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)
  ## Encoding city
  if not multiple_models:
    dataset['city'] = dataset['city'].apply(lambda x: 1 if x == 'sj' else 0)

  return dataset


def feature_engineer_pipeline(dataset, multiple_models=False):
  """
  This function is to aggregrate all feature engineering
  """
  dataset = remove_features(dataset)
  dataset = feature_crossing(dataset)
  dataset = other_preprocess(dataset, multiple_models)

  return dataset