# House-Prices-Advanced-Regression-Techniques
Joanna Broniarek

**The goal of this repository was to provide an analysis for Kaggle's competition:  https://www.kaggle.com/c/house-prices-advanced-regression-techniques**

![kaggle-image](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

**My best score on the Kaggle Leaderboard: 0.11329**

## Description

**EDA and Data tidying:**

1. Removing columns that contain the same value in 100% ["Street", "Utilities"]
2. Removing outliers : GrLivArea more than 4500.
3. Improving values like Year more than 2017.
4. Handling missing numerical values:
  + LotFrontage according to median in specific Neighborhood
  + With constant = 0 for :
['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]
  + The rest of numerical columns (apart from point 5) with median.
5. Transformation of some numerical features that are actually categorical:
['MSSubClass', 'OverallCond’]
6. Handling missing categorical values. (specific for each feature)
7. Transformation of skewed features:
  + SalePrice – log transformation
  + Other features with skeweness > 0.5 using BoxCox transformation
  + Transformation some categorical features (with specific order) into numerical

**Feature Engineering:**

1. Feature Isgarage defined according to feature GarageArea (1 – if more than 0)
2. Feature Isfireplace defined according to feature Fireplaces (if more than 0)
3. Feature Ispool defined according to feature PoolArea (if more than 0)
4. Feature Issecondfloor defined according to feature 2ndFlrSF (if more than 0)
5. Feature IsOpenPorch defined according to feature OpenPorchSF (if more than 0)
6. Feature IsWoodDeck defined according to feature WoodDeckSF (if more than 0)
7. Feature TotalSqrtFeet defined as sum of GrLivArea and TotalBsmtSF
8. Feature TotalBaths defined as BsmtFullBath + FullBath + BsmtHalfBath/2 + HalfBath/2.
9. Feature Neighborhood (transformation into 0, 1, 2) according to statistics if specific Neighborhood is rather rich/poor or between them.
10. One-Hot Encoding for categorical data

**Modelization:**

Scaling - RobustScaler
1. Linear Regression
2. LASSO model selection
3. GradientBoostingRegressor
4. XGBRegressor
5. ElasticNet
6. LGBMRegressor
7. BaggingRegressor

**Training:**
1. StackingCVRegressor on models: [Lasso, ElasticNet, XGB, LGBM]
2. Weighted predictions 0.2*ElasticNet + 0.25*lasso + 0.15*LGBM + 0.4*StackedModels

## Environment specification:

* python 3.6.4
* numpy 1.14.2
* scipy 1.1.0rc1
* seaborn 0.9.0
* sklearn 0.20.1
* pandas 0.22.0
* sklearn 0.20.1
* xgboost 0.72
* lightgbm 2.2.2
