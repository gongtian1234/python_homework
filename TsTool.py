# -*- coding: utf-8 -*-
import pandas as pd ;
import matplotlib.pylab as plt;
import statsmodels.api as sm;
from statsmodels.tsa.stattools import adfuller;
from statsmodels.tsa.stattools import kpss;
from statsmodels.graphics.api import qqplot;


'''
# pip install pmdarima
# https://www.alkaline-ml.com/pmdarima/
#https://blog.csdn.net/m0_37700507/article/details/84855235
'''

def plot_df(df, x, y, title="", xlabel='date', ylabel='price', dpi=100):
    '''
    画出时间序列图
    :param df:
    :param x:
    :param y:
    :param title:
    :param xlabel:
    :param ylabel:
    :param dpi:
    :return:
    '''
    plt.figure(figsize=(16,5), dpi=dpi);
    plt.plot(x, y, color='tab:red');
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel);
    plt.show();

def testStationarity( ts ):
    '''
    提供逐个参数的稳定性的方式
    下面是不同检验结果的判断方式，请注意：
    Case 1: Both tests conclude that the series is not stationary -> series is not stationary
    Case 2: Both tests conclude that the series is stationary -> series is stationary
    Case 3: KPSS = stationary and ADF = not stationary  -> trend stationary, remove the trend to make series strict stationary
    Case 4: KPSS = not stationary and ADF = stationary -> difference stationary, use differencing to make series stationary
    :param df:
    :return:
    '''
    # Perform Dickey-Fuller test:
    # the null hypothesis that a unit root is present in a time series sample
    # The alternative hypothesis is  is usually stationarity or trend-stationarity
    print('--------------------------Dickey-Fuller test-----------------------------------');
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value ;
    print(dfoutput)

    # Perform Kwiatkowski-Phillips-Schmidt-Shin test for stationarity:
    # The null hypothesis for the test is that the data is stationary.
    # The alternate hypothesis for the test is that the data is not stationary.
    print('--------------------------kpss test-----------------------------------');
    (kpss_stat, p_value, lags, crit)= kpss(ts,lags=2 ) ;
    dftest = [kpss_stat, p_value, lags];
    dfoutput = pd.Series(dftest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])
    for key, value in crit.items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def drawAcfPacf( ts ):
    '''
    画出Acf和Pacf, 以协助判断ARIMA的参数;
    :param timeseries:
    :return:
    '''
    fig = plt.figure(figsize=(12, 8));
    ax1 = fig.add_subplot(211);
    fig = sm.graphics.tsa.plot_acf(ts, lags=40, ax=ax1);
    ax2 = fig.add_subplot(212);
    fig = sm.graphics.tsa.plot_pacf(ts, lags=40, ax=ax2);
    plt.title("acf and pacf");
    plt.show();



def tsdiag( resid ):
    '''
    展示模型检验的结果
    :param resid:
    :return:
    '''
    fig = plt.figure(figsize=(12,8));

    ax1 = fig.add_subplot(311);
    fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1);

    ax2 = fig.add_subplot(312);
    fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2);

    ax3 = fig.add_subplot(313);
    fig = qqplot(resid, line='q', ax=ax3, fit=True)

    plt.show();





