import pandas as pd
import numpy as np
from get_data import get_rdk_data, read_parquet_tick_data
import warnings
import scipy.stats as stats
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def cal_BVC_vol(df_tick):
    df_tick['LastPrice_diff'] = df_tick['last'].diff()
    # 按照TradingDate和min_time列进行分组，并计算每组的LastPrice_diff列的取值的标准差
    df_tick['diff_sigma'] = df_tick.groupby(['trading_date', 'min_time'])['LastPrice_diff'].transform('std')
    df_tick['ratio'] = df_tick['LastPrice_diff'] / df_tick['diff_sigma']
    df_tick['cdf_value'] = df_tick['ratio'].apply(lambda x: stats.norm.cdf(x))
    df_tick['buy_vol'] = df_tick['volume'] * df_tick['cdf_value']
    df_tick['sell_vol'] = df_tick['volume'] - df_tick['buy_vol']
    return df_tick


def cal_sigma_est(col):
    mean_bar = col.sum()/(len(col)-1)
    sigma = np.sqrt(((col-mean_bar)**2).sum()/len(col))
    return sigma


def sell_cal_r_est(group):
    group['sell_r_est'] = (((group['sell_vol']-group['sell_mean_est'])/group['sell_sigma_est'])**3).sum()/len('sell_vol')
    return group['sell_r_est']


def buy_cal_r_est(group):
    group['buy_r_est'] = (((group['buy_vol']-group['buy_mean_est'])/group['buy_sigma_est'])**3).sum()/len('buy_vol')
    return group['sell_r_est']


def cal_MEPIN(df_min,direction):
    df_min[direction+'_up'] = 2*(df_min[direction+'_sigma']**2 - df_min[direction+'_mean'])**2
    df_min[direction+'_down11'] = 4*(df_min[direction+'_sigma']**2 - df_min[direction+'_mean'])**3
    df_min[direction+'_down2'] = (df_min[direction + '_r']*df_min[direction + '_sigma']**3) - 3*df_min[direction + '_sigma']**2 + 2*df_min[direction+'_mean']
    df_min[direction+'_down1'] = np.sqrt(df_min[direction+'_down11'] + df_min[direction+'_down11']**2)
    df_min[direction+'_down'] = df_min[direction+'_down1'] + df_min[direction+'_down2']
    df_min[direction+'_MEPIN'] = (df_min[direction+'_up']/df_min[direction+'_down'])/df_min[direction+'_mean']
    return df_min


def cal_MEPIN_min(df_min, df_tick):
    df_tick = cal_BVC_vol(df_tick)
    # Sell Side
    df_tick['sell_mean_est'] = df_tick.groupby(['min_time', 'trading_date'])['sell_vol'].transform('mean')
    df_tick['sell_sigma_est'] = df_tick.groupby(['min_time', 'trading_date'])['sell_vol'].transform(cal_sigma_est)
    df_tick['sell_r_est'] = df_tick.groupby(['min_time', 'trading_date']).apply(sell_cal_r_est).reset_index(level=[0, 1], drop=True)
    sell_mean_dict = df_tick.groupby(['min_time', 'trading_date'])['sell_mean_est'].mean().to_dict()
    sell_sigma_dict = df_tick.groupby(['min_time', 'trading_date'])['sell_sigma_est'].mean().to_dict()
    sell_r_dict = df_tick.groupby(['min_time', 'trading_date'])['sell_r_est'].mean().to_dict()

    # Buy Side
    df_tick['buy_mean_est'] = df_tick.groupby(['min_time', 'trading_date'])['buy_vol'].transform('mean')
    df_tick['buy_sigma_est'] = df_tick.groupby(['min_time', 'trading_date'])['buy_vol'].transform(cal_sigma_est)
    df_tick['buy_r_est'] = df_tick.groupby(['min_time', 'trading_date']).apply(buy_cal_r_est).reset_index(level=[0, 1], drop=True)
    buy_mean_dict = df_tick.groupby(['min_time', 'trading_date'])['buy_mean_est'].mean().to_dict()
    buy_sigma_dict = df_tick.groupby(['min_time', 'trading_date'])['buy_sigma_est'].mean().to_dict()
    buy_r_dict = df_tick.groupby(['min_time', 'trading_date'])['buy_r_est'].mean().to_dict()

    # min
    df_min['sell_mean'] = df_min.apply(lambda x: sell_mean_dict.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['sell_sigma'] = df_min.apply(lambda x: sell_sigma_dict.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['sell_r'] = df_min.apply(lambda x: sell_r_dict.get((x['min_time'], x['trading_date']), 0), axis=1)

    df_min['buy_mean'] = df_min.apply(lambda x: buy_mean_dict.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['buy_sigma'] = df_min.apply(lambda x: buy_sigma_dict.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['buy_r'] = df_min.apply(lambda x: buy_r_dict.get((x['min_time'], x['trading_date']), 0), axis=1)

    df_min = cal_MEPIN(df_min,'buy')
    df_min = cal_MEPIN(df_min,'sell')
    df_min['MEPIN_min'] = (df_min['buy_MEPIN'] + df_min['sell_MEPIN'])/2
    df_min['MEPIN_min'] = df_min['MEPIN_min'].ffill()

    return df_min


if __name__ == '__main__':
    data = get_rdk_data('AG888','2024-01-01','2024-10-25','E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data','min')
    # 给定起始日期和截止日期
    start_date = '2020-01-01 09:00:00'
    end_date = '2024-10-25 15:30:00'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # 筛选出日期索引处于给定区间中的那部分数据
    filtered_df = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    filtered_df = filtered_df.rename(columns={'date':'trading_date'})
    filtered_df['trading_date'] = pd.to_datetime(filtered_df['trading_date'])
    filtered_df['min_time'] = filtered_df.index
    filtered_df['min_time'] = pd.to_datetime(filtered_df['min_time'])
    # 读取tick数据
    data_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\ticks\\AG88.parquet'
    tick_df = pd.read_parquet(data_path)
    tick_df['trading_date'] = pd.to_datetime(tick_df['trading_date'])
    tick_df = tick_df[(tick_df['trading_date'] >= start_date) & (tick_df['trading_date'] <= end_date)]
    tick_df['min_time'] = tick_df.index
    tick_df['min_time'] = pd.to_datetime(tick_df['min_time'])
    tick_df['min_time'] = tick_df['min_time'].dt.floor('T')
    tick_df['min_time'] = pd.to_datetime(tick_df['min_time']) + pd.Timedelta(minutes=1)

    # 因子计算
    filtered_df= cal_MEPIN_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['MEPIN_min'])
    filtered_df = filtered_df.drop(columns='MEPIN_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['MEPIN_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'MEPIN_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")

    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()


