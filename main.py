import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import seaborn as sns

from scipy.stats import pearsonr

import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.kernel_ridge as kernel_ridge
import sklearn.ensemble as ensemble
import sklearn.neural_network as nn
import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = 'Arial'
plt.style.use('seaborn-poster')


def load_data(train_path, test_path):
    print('\nLoad data.')
    train_data = pd.read_csv(train_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    print('Number of train data:', len(train_data))
    print('Number of test data:', len(test_data))
    return train_data, test_data


def categorize(data):
    print('\nCatergorize quantitative and qualitative variables.')
    quantitative = [f for f in data.columns if data.dtypes[f] != 'object']
    quantitative.remove('SalePrice')
    qualitative = [f for f in data.columns if data.dtypes[f] == 'object']
    return quantitative, qualitative


def check_missing(data):
    print('\nCheck missing data.')
    plt.figure(figsize=(8, 6), dpi=160)
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot(kind='bar')
    plt.title('Number of missing values')
    plt.tick_params(axis='x', labelsize=14)

    plt.subplots_adjust(bottom=0.3)
    plt.savefig('img/check_missing.png')
    plt.close()


def drop_missing(train_data, test_data):
    print('\nDrop variables with over 50% missing values.')
    missing = train_data.isnull().sum()
    missing = missing[missing > 0]
    missing_variables = []
    for i in range(len(missing)):
        if missing.values[i] > 0.5 * len(train_data):
            missing_variables.append(missing.index[i])
    train_data.drop(columns=missing_variables, inplace=True)
    test_data.drop(columns=missing_variables, inplace=True)


def plot_distribution(data):
    print('\nPlot distribution of "SalePrice".')
    y = data['SalePrice']
    plt.figure(figsize=(8, 18), dpi=160)

    ax1 = plt.subplot(311)
    ax1.set_title('Normal')
    sns.distplot(y, kde=False, fit=stats.norm)
    ax1.ticklabel_format(axis='both', style='sci', scilimits=(-4, 4), useMathText=True)

    ax2 = plt.subplot(312)
    ax2.set_title('Log Normal')
    sns.distplot(y, kde=False, fit=stats.lognorm)
    ax2.ticklabel_format(axis='both', style='sci', scilimits=(-4, 4), useMathText=True)

    ax3 = plt.subplot(313)
    ax3.set_title('Johnson SU')
    sns.distplot(y, kde=False, fit=stats.johnsonsu)
    ax3.ticklabel_format(axis='both', style='sci', scilimits=(-4, 4), useMathText=True)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3)
    plt.savefig('img/distribution.png')
    plt.close()


def johnson(x):
    gamma, eta, epsilon, lbda = stats.johnsonsu.fit(x)
    x = gamma + eta * np.arcsinh((x - epsilon) / lbda)
    return x, gamma, eta, epsilon, lbda


def johnson_inverse(x, gamma, eta, epsilon, lbda):
    return lbda * np.sinh((x - gamma) / eta) + epsilon


def check_normality(data, alpha=0.05):
    print('\nCheck normality of SalePrice.')
    _, p_norm = stats.normaltest(data['SalePrice'])
    _, p_lognorm = stats.normaltest(np.log(data['SalePrice']))
    _, p_johnson = stats.normaltest(johnson(data['SalePrice'])[0])
    print('Normality: {}\talpha: {:.2f}\tp-value: {:.4f}'.format(
        str(p_norm > alpha), alpha, p_norm))
    print('Log Normality: {}\talpha: {:.2f}\tp-value: {:.4f}'.format(
        str(p_lognorm > alpha), alpha, p_lognorm))
    print('Johnson SU: {}\talpha: {:.2f}\tp-value: {:.4f}'.format(
        str(p_johnson > alpha), alpha, p_johnson))


def plot_quantitatives(data, features):
    print('\nPlot facetgrid of quantitative variables.')
    plt.figure(dpi=160)
    f = pd.melt(data, value_vars=features)
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False, size=4)
    g = g.map(sns.distplot, "value")
    
    plt.savefig('img/quantitatives.png')
    plt.close()


def _boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x = plt.xticks(rotation=90)


def plot_qualitatives(data, features):
    print('\nPlot boxplot of qualitative variables.')
    plt.figure(dpi=160)
    f = pd.melt(data, id_vars=['SalePrice'], value_vars=features)
    g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False, size=4)
    g = g.map(_boxplot, 'value', 'SalePrice')

    plt.savefig('img/qualitatives.png')
    plt.close()


def encode_qualitatives(train_data, test_data, features):
    print('\nEncode qualitative variables to discrete quantitative variables.')
    for q in features:
        ordering = pd.DataFrame()
        ordering['val'] = train_data[q].unique()
        ordering.index = ordering.val
        ordering['spmean'] = train_data[[q, 'SalePrice']].groupby(q).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
        ordering['ordering'] = range(1, ordering.shape[0] + 1)
        ordering = ordering['ordering'].to_dict()
        for cat, o in ordering.items():
            train_data.loc[train_data[q] == cat, q] = o
            test_data.loc[test_data[q] == cat, q] = o
    
    return train_data, test_data


def plot_correlation(data, features):
    print('\nPlot correlation heatmap of the variables.')
    corr = data[features].corr()
    print(data[features].head())
    print(corr.head())
    plt.figure(figsize=(20, 15), dpi=160)
    ax = sns.heatmap(corr, cmap=mpl.colors.ListedColormap(plt.cm.seismic(np.linspace(0.1, 0.9, 8))), 
                     cbar=False, vmin=-1, vmax=1, xticklabels=features, yticklabels=features)
    ax.set_xticklabels(features, fontsize=12)
    ax.set_yticklabels(features, fontsize=12)
    cb = plt.colorbar(ax.collections[0], aspect=40)
    cb.ax.tick_params(labelsize=20)
    cb.set_ticks(np.linspace(-1, 1, 9))
    cb.set_ticklabels(np.linspace(-1, 1, 9))
    cb.set_label('pearson correlation', fontsize=24)

    plt.subplots_adjust(right=0.95, top=0.95)
    plt.savefig('img/correlation.png')
    plt.close()


def _pairplot(x, y, **kwargs):
    ax = plt.gca()
    ts = pd.DataFrame({'time': x, 'val': y})
    ts = ts.groupby('time').mean()
    ts.plot(ax=ax)


def plot_paircorrelation(data, features):
    print('\nPlot paircorrelation figures between the variables and "SalePrice".')
    plt.figure(dpi=160)
    f = pd.melt(data, id_vars=['SalePrice'], value_vars=features)
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False, size=4)
    g = g.map(_pairplot, 'value', 'SalePrice')
    
    plt.savefig('img/paircorrelation.png')
    plt.close()


def divergence(p, q):
    M = (p + q) / 2
    KLD = stats.entropy(p, q)
    JSD = 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)
    return KLD, JSD


def evaluate(Y, Y_pred):
    metrics = ['R2', 'RMSLE', 'MAE', 'MAPE', 'KLD', 'JSD']
    R2 = r2_score(Y, Y_pred)
    MAE = mean_absolute_error(Y, Y_pred)
    MAPE = mean_absolute_percentage_error(Y, Y_pred)
    RMSLE = np.sqrt(mean_squared_error(np.log(Y), np.log(Y_pred)))
    KLD, JSD = divergence(Y, Y_pred)
    print('R2: {:.4f}'.format(R2))
    print('MAE: {:.0f}'.format(MAE))
    print('MAPE: {:.2f} %'.format(100 * MAPE))
    print('RMSLE: {:.4f}'.format(RMSLE))
    print('KLD: {:.6f}'.format(KLD))
    print('JSD: {:.6f}'.format(JSD))
    scores = {}
    for m in metrics:
        scores[m] = locals()[m]
    return scores


def save_scores(scores, method='linear'):
    df = pd.DataFrame(data=scores, columns=scores.keys(), index=[0])
    df.to_csv('score/{}.csv'.format(method), float_format='%.6f', index=None)


def save_best_params(best_params, method='linear'):
    df = pd.DataFrame(data=best_params, columns=best_params.keys(), index=[0])
    df.to_csv('score/{}_best.csv'.format(method), index=None)


def plot_regression(results, scores, method='linear', title='Linear Regression'):
    # Distribution approximation
    plt.figure(figsize=(6, 6), dpi=160)
    ax = sns.histplot(data=results, kde=True)
    ax.set_xlim([0, 1e6])
    ax.set_ylim([0, 200])
    ax.text(7e5, 120, 'KLD: {:.6f}\nJSD: {:.6f}'.format(scores['KLD'], scores['JSD']), 
            fontsize=12, linespacing=1.5,
            bbox={'facecolor': 'w', 'edgecolor': 'grey', 'pad': 5})

    ax.ticklabel_format(axis='both', style='sci',
                        scilimits=(-4, 4), useMathText=True)
    ax.set_title(title)
    
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('img/{}_dist.png'.format(method))
    plt.close()

    # x-y approximation
    plt.figure(figsize=(6, 6), dpi=160)
    ax = sns.scatterplot(data=results, x='truth', y='pred', s=16, color='b')
    ax.set_xlim([0, 1e6])
    ax.set_ylim([0, 1e6])
    ax.text(7e5, 8e5, 'R$^2$: {:.4f}\nMAE: {:.0f}\nMAPE: {:.2f} %\nRMSLE: {:.4f}'
            .format(scores['R2'], scores['MAE'], 100 * scores['MAPE'], scores['RMSLE']),
            fontsize=12, linespacing=1.5,
            bbox={'facecolor': 'w', 'edgecolor': 'grey', 'pad': 5})
    ax.ticklabel_format(axis='both', style='sci',
                        scilimits=(-4, 4), useMathText=True)
    ax.plot(np.linspace(0, 1e6, 100), np.linspace(0, 1e6, 100), color='r', linewidth=1)
    ax.set_title(title)
    
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('img/{}_xy.png'.format(method))
    plt.close()


def fill_nan(train_data, test_data, features):
    print('\nFill nan')
    # Fill missing data as median
    for col in features:
        train_data[col] = train_data[col].fillna(train_data[col].median())
        test_data[col] = test_data[col].fillna(train_data[col].median())
    
    return train_data, test_data


def create_new_features(train_data, test_data):
    print('\nCreate new features.')
    for data in [train_data, test_data]:
        data['TotalHouseArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
        data['YearsSinceRemodel'] = data['YrSold'].astype(int) - data['YearRemodAdd'].astype(int)
        data['Total_Home_Quality'] = data['OverallQual'].astype(int) + data['OverallCond'].astype(int)
        data['HasWoodDeck'] = (data['WoodDeckSF'] == 0) * 1
        data['HasOpenPorch'] = (data['OpenPorchSF'] == 0) * 1
        data['HasEnclosedPorch'] = (data['EnclosedPorch'] == 0) * 1
        data['Has3SsnPorch'] = (data['3SsnPorch'] == 0) * 1
        data['HasScreenPorch'] = (data['ScreenPorch'] == 0) * 1
        data["TotalAllArea"] = data['TotalHouseArea'] + data['GarageArea']
        data["TotalHouse_and_OverallQual"] = data['TotalHouseArea'] * data['OverallQual']
        data["GrLivArea_and_OverallQual"] = data['GrLivArea'] * data['OverallQual']
        data["LotArea_and_OverallQual"] = data['LotArea'] * data['OverallQual']
        data["MSZoning_and_TotalHouse"] = data['MSZoning'] * data['TotalHouseArea']
        data["MSZoning_and_OverallQual"] = data['MSZoning'] + data['OverallQual']
        data["MSZoning_and_YearBuilt"] = data['MSZoning'] + data['YearBuilt']
        data["Neighborhood_and_TotalHouse"] = data['Neighborhood'] * data['TotalHouseArea']
        data["Neighborhood_and_OverallQual"] = data['Neighborhood'] + data['OverallQual']  
        data["Neighborhood_and_YearBuilt"] = data['Neighborhood'] + data['YearBuilt']
        data["BsmtFinSF1_and_OverallQual"] = data['BsmtFinSF1'] * data['OverallQual']
        data["Functional_and_TotalHouse"] = data['Functional'] * data['TotalHouseArea']
        data["Functional_and_OverallQual"] = data['Functional'] + data['OverallQual']
        data["TotalHouse_and_LotArea"] = data['TotalHouseArea'] + data['LotArea']
        data["Condition1_and_TotalHouse"] = data['Condition1'] * data['TotalHouseArea']
        data["Condition1_and_OverallQual"] = data['Condition1'] + data['OverallQual']
        data["Bsmt"] = data["BsmtFinSF1"] + data['BsmtFinSF2'] + data['BsmtUnfSF']
        data["Rooms"] = data["FullBath"]+data["TotRmsAbvGrd"]
        data["PorchArea"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
        data["TotalPlace"] = data["TotalAllArea"] + data["PorchArea"]
    
    all_features = list(test_data.keys())
    
    return train_data, test_data, all_features


def find_major_factors(train_data, test_data, features):
    print('\nFind major factors using pearson correlations.')
    plt.figure(figsize=(10, 6), dpi=160)
    plt.style.use('seaborn-notebook')
    pearson_corr = []
    for f in features:
        pearson_corr.append(pearsonr(train_data[f], train_data['SalePrice'])[0])
    df = pd.DataFrame(data=np.array(pearson_corr).reshape((1, -1)), columns=features, index=[0])
    df = df.sort_values(by=[0], axis=1, ascending=False)
    df1 = df.iloc[:, 0: 40]
    df2 = df.iloc[:, 40:]
    
    ax1 = plt.subplot(211)
    sns.barplot(data=df1, ax=ax1)
    plt.xticks(rotation=90, fontsize=7)

    ax2 = plt.subplot(212)
    sns.barplot(data=df2, ax=ax2)
    plt.xticks(rotation=90, fontsize=7)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25, hspace=0.5)
    plt.savefig('img/pearson.png')
    plt.close()

    major_factors = features
    for f in features:
        if np.abs(np.float(df[f])) < 0.2:
            train_data = train_data.drop(f, axis=1)
            test_data = test_data.drop(f, axis=1)
            major_factors.remove(f)
    
    return train_data, test_data, major_factors


def get_values(train_data, test_data, features):
    X, Y = train_data[features].values, train_data['SalePrice'].values
    X_test = test_data[features].values
    return X, Y, X_test


def feature_combination(X, X_test):
    print('\nPolynomial feature combination')
    polynomial = PolynomialFeatures(2, interaction_only=True).fit(X)
    X = polynomial.transform(X)
    X_test = polynomial.transform(X_test)
    return X, X_test


def dimension_reduction(X, X_test, n_components=100):
    print('\nPrincipal Component Analysis')
    pca = PCA(n_components).fit(X)
    X = pca.transform(X)
    X_test = pca.transform(X_test)
    return X, X_test


def scaling(X, X_test):
    print('\nStandard scaling')
    scalar = RobustScaler().fit(X)
    X = scalar.transform(X)
    X_test = scalar.transform(X_test)
    return X, X_test


def regression(X, Y, X_test, method, model, param_grid, title):
    print('\n{}'.format(title))

    # Grid search cross validation
    grid_search_cv = GridSearchCV(model, param_grid, cv=10, n_jobs=-1)
    grid_search_cv.fit(X, np.log(Y))

    print('\nBest params:', grid_search_cv.best_params_)
    Y_pred = np.exp(grid_search_cv.predict(X))
    
    # Evaluation
    scores = evaluate(Y, Y_pred)

    # Plot results
    results = pd.DataFrame(data=np.array([Y, Y_pred]).T, columns=['truth', 'pred'])
    plot_regression(results, scores, method, title)
    
    # Save scores and best params
    save_scores(scores, method)
    if method == 'mlp':
        best_params = pd.DataFrame(data=grid_search_cv.best_params_,
                                   columns=grid_search_cv.best_params_.keys())
    else:
        best_params = pd.DataFrame(data=grid_search_cv.best_params_,
                                   columns=grid_search_cv.best_params_.keys(), index=[0])
    save_best_params(best_params, method)

    # Give final prediction
    test_pred = np.exp(grid_search_cv.predict(X_test))
    
    return test_pred


def linear_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='linear',
                           model=linear_model.LinearRegression(n_jobs=-1), 
                           param_grid={'fit_intercept': [True, False]}, 
                           title='Linear Regression')
    
    return test_pred


def lasso_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='lasso',
                           model=linear_model.Lasso(max_iter=1e4),
                           param_grid={'alpha': np.logspace(-6, 0, 7)},
                           title='Lasso Regression')

    return test_pred


def ridge_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='ridge',
                           model=linear_model.Ridge(max_iter=1e4),
                           param_grid={'alpha': np.logspace(-6, 0, 7)},
                           title='Ridge Regression')

    return test_pred


def huber_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='huber',
                           model=linear_model.HuberRegressor(max_iter=1e4, tol=1e-3),
                           param_grid={'alpha': np.logspace(-6, -2, 5)},
                           title='Huber Regression')

    return test_pred


def kernel_ridge_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='kernel_ridge',
                           model=kernel_ridge.KernelRidge(),
                           param_grid={'alpha': np.logspace(-2, 2, 5),
                                       'kernel': ['linear', 'rbf']},
                           title='Kernel Ridge Regression')

    return test_pred


def svm_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='svm',
                           model=svm.SVR(),
                           param_grid={'C': np.logspace(-2, 2, 5),
                                       'epsilon': np.logspace(-6, -2, 5),
                                       'gamma': ['scale', 'auto']},
                           title='SVM Regression')

    return test_pred


def random_forest_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='random_forest',
                           model=ensemble.RandomForestRegressor(),
                           param_grid={'max_depth': [50, 100, 200],
                                       'n_estimators': [100, 1000]},
                           title='Random Forest Regression')

    return test_pred


def bagging_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='bagging',
                           model=ensemble.RandomForestRegressor(),
                           param_grid={'n_estimators': [10, 100, 1000]},
                           title='Bagging Regression')

    return test_pred


def adaboost_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='adaboost',
                           model=ensemble.AdaBoostRegressor(),
                           param_grid={'n_estimators': [100, 200, 500],
                                       'loss': ['linear', 'square']},
                           title='Adaboost Regression')

    return test_pred
 

def xgboost_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='xgboost',
                           model=xgb.XGBRegressor(alpha=1.5),
                           param_grid={'n_estimators': [100, 200, 500]},
                           title='XGboost Regression')

    return test_pred


def lightgbm_regression(X, Y, X_test):
    test_pred = regression(X, Y, X_test, method='lightgbm',
                           model=lgbm.LGBMRegressor(),
                           param_grid={'num_leaves': [30, 50, 100],
                                       'n_estimators': [50, 100, 200]},
                           title='LightGBM Regression')

    return test_pred


def mlp_regression(X, Y, X_test):
    scalar = MinMaxScaler((-1, 1)).fit(X)
    X = scalar.transform(X)
    X_test = scalar.transform(X_test)
    test_pred = regression(X, Y, X_test, method='mlp',
                           model=nn.MLPRegressor(max_iter=10000, alpha=1e-4),
                           param_grid={'hidden_layer_sizes': [(128, ) * n for n in range(1, 4)]},
                           title='MLP Regression')

    return test_pred


def compare_methods(scores_path, save_path):
    scores = []
    methods = []
    for filename in os.listdir(scores_path):
        if 'best' not in filename:
            methods.append(filename.replace('.csv', ''))
            scores.append(pd.read_csv(os.path.join(scores_path, filename)).values)
            columns = pd.read_csv(os.path.join(scores_path, filename)).columns

    scores_df = pd.DataFrame(data=np.array(scores).squeeze(), columns=columns, index=methods)
    scores_df.to_csv(save_path, index=methods)

    return scores_df


def give_final_prediction(test_data, test_pred, scores_df, test_pred_path, best_path):
    test_pred_df = pd.DataFrame(data=test_pred, columns=test_pred.keys(), index=test_data.index)
    test_pred_df.to_csv(test_pred_path)
    best_method = scores_df.sort_values(by=['RMSLE']).index[0]
    best_test_pred = pd.DataFrame(test_pred_df[best_method])
    best_test_pred.index = test_data.index
    best_test_pred.set_axis(['SalePrice'], axis='columns', inplace=True)
    best_test_pred.to_csv(best_path)
    print('\nThe best method is {}'.format(best_method))


if __name__ == '__main__':

    ### Data overview and processing ###
    # Load data for training and test
    train_data, test_data = load_data('data/train.csv', 'data/test.csv')

    # Check and drop missing data
    check_missing(train_data)
    drop_missing(train_data, test_data)

    # Categorize quantitative and qualitative variables
    quantitatives, qualitatives = categorize(train_data)

    # Check distribution of the variables 
    plot_distribution(train_data)
    check_normality(train_data)
    plot_quantitatives(train_data, quantitatives)
    plot_qualitatives(train_data, qualitatives)

    # Encode qualitative variables to quantitative ones
    train_data, test_data = encode_qualitatives(train_data, test_data, qualitatives)
    features = quantitatives + qualitatives

    # Check the correlation between the variables
    train_data, test_data = fill_nan(train_data, test_data, features)
    plot_correlation(train_data, features + ['SalePrice'])
    plot_paircorrelation(train_data, features)

    # feature combination and dimension_reduction
    train_data, test_data, all_features = create_new_features(train_data, test_data)
    train_data, test_data, major_factors = find_major_factors(train_data, test_data, all_features)
    X, Y, X_test = get_values(train_data, test_data, major_factors)
    X, X_test = feature_combination(X, X_test)
    X, X_test = dimension_reduction(X, X_test, n_components=200)
    X, X_test = scaling(X, X_test)

    ### Regression ###
    # Regression with diversed models
    test_pred = {}
    test_pred['linear'] = linear_regression(X, Y, X_test)
    test_pred['lasso'] = lasso_regression(X, Y, X_test)
    test_pred['ridge'] = ridge_regression(X, Y, X_test)
    test_pred['huber'] = huber_regression(X, Y, X_test)
    test_pred['kernel_ridge'] = kernel_ridge_regression(X, Y, X_test)
    test_pred['svm'] = svm_regression(X, Y, X_test)
    test_pred['random_forest'] = random_forest_regression(X, Y, X_test)
    test_pred['bagging'] = bagging_regression(X, Y, X_test)
    test_pred['adaboost'] = adaboost_regression(X, Y, X_test)
    test_pred['xgboost'] = xgboost_regression(X, Y, X_test)
    test_pred['lightgbm'] = lightgbm_regression(X, Y, X_test)
    test_pred['mlp'] = mlp_regression(X, Y, X_test)
    
    # Compare the models
    scores_df = compare_methods('./score', 'comparison.csv')
    
    # Give final prediction on the test set
    give_final_prediction(test_data, test_pred, scores_df, 'test_pred.csv', 'submission.csv')
    
