import pandas as pd
import numpy as np
from get_data import get_rdk_data, read_parquet_tick_data
import warnings
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def cal_CS_beta(df_min):
    df_min['HL'] = df_min['high'] / df_min['low']
    df_min['HL'] = df_min['HL'].apply(lambda x: np.log(x))
    df_min['Prev_HL'] = df_min['HL'].shift(1)
    df_min['beta'] = (df_min['HL']**2 + df_min['Prev_HL']**2)/2

    return df_min


def cal_CS_gamma(df_min):
    df_min['Max_HighPrice'] = df_min[['high']].shift(1).fillna(0)
    df_min['Max_HighPrice'] = df_min[['high', 'Max_HighPrice']].max(axis=1)
    df_min['Min_LowPrice'] = df_min[['low']].shift(1).fillna(0)
    df_min['Min_LowPrice'] = df_min[['low', 'Min_LowPrice']].min(axis=1)
    df_min['MaxH_MinL'] = df_min['Max_HighPrice'] / df_min['Min_LowPrice']
    df_min['MaxH_MinL'] = df_min['MaxH_MinL'].apply(lambda x: np.log(x))
    df_min['gamma'] = df_min['MaxH_MinL']**2

    return df_min


def cal_CS_alpha(df_min):
    df_min = cal_CS_beta(df_min)
    df_min = cal_CS_gamma(df_min)

    df_min['doublebeta'] = 2*df_min['beta']
    df_min['doublebeta'] = df_min['doublebeta'].apply(lambda x: np.sqrt(x))
    df_min['beta'] = df_min['beta'].apply(lambda x: np.sqrt(x))
    df_min['alpha1'] = (df_min['doublebeta'] - df_min['beta'])/(3-2*np.sqrt(2))
    df_min['alpha2'] = df_min['gamma'] / (3-2*np.sqrt(2))
    df_min['alpha2'] = df_min['alpha2'].apply(lambda x: np.sqrt(x))
    df_min['alpha'] = df_min['alpha1']-df_min['alpha2']

    return df_min


def cal_CS_spread_min(df_min):
    df_min = cal_CS_alpha(df_min)
    df_min['alpha'] = df_min['alpha'].apply(lambda x: np.exp(x))
    df_min['spread_CS_min'] = 2*(df_min['alpha']-1)/(df_min['alpha']+1)
    df_min['spread_CS_min'] = df_min['spread_CS_min'].ffill()

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
    # 因子计算
    filtered_df = cal_CS_spread_min(filtered_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['spread_CS_min'])
    filtered_df = filtered_df.drop(columns='spread_CS_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['spread_CS_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'spread_CS_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()

