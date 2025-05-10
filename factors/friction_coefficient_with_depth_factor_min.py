import pandas as pd
import numpy as np
from get_data import get_rdk_data
import warnings
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')



# With Depth
def calc_illiq_fraction(row):
    up = row['volume'] ** 2
    if row['close'] > row['previous_close']:

        dr = (row['open_interest'] ** 2) * (
                np.abs((row['high'] - row['previous_close']) / (
                            row['previous_close'] + 0.001)) + np.abs(
            (row['low'] - row['high']) / (row['high'] + 0.001)) + np.abs(
            (row['close'] - row['low']) / (row['low'] + 0.001)))

        fac_value = up / dr
    elif row['close'] < row['previous_close']:

        dr = -1 * ((row['open_interest'] ** 2) * (
                np.abs((row['low'] - row['previous_close']) / (
                            row['previous_close'] + 0.001)) + np.abs(
            (row['high'] - row['low']) / (row['low'] + 0.001)) + np.abs(
            (row['close'] - row['high']) / (row['high'] + 0.001))))

        fac_value = up / dr
    else:
        fac_value = np.nan
    return fac_value


def fraction_coefficient_min(df):
    df['previous_close'] = df['close'].shift(1)
    df['previous_close'] = df['previous_close'].fillna(0)
    df['turnover_ratio'] = df['volume'] / df['open_interest']

    df['Factor'] = df.apply(lambda row: calc_illiq_fraction(row), axis=1)
    df['Factor'] = df['Factor'].replace([np.inf, -np.inf], np.nan)
    df['ffilled_factor'] = df['Factor'].ffill()
    df['Factor'] = df['Factor'].fillna((df['ffilled_factor'] + df['turnover_ratio']) / 2)
    df['Factor_MA'] = df['Factor'].rolling(window=10, min_periods=1).mean()
    df = df.drop('ffilled_factor', axis=1)
    return df['Factor_MA'].values


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
    filtered_df['friction_coefficient_depth_min'] = fraction_coefficient_min(filtered_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['friction_coefficient_depth_min'])
    filtered_df = filtered_df.drop(columns='friction_coefficient_depth_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)


    factor_df = pd.DataFrame(merged_df['friction_coefficient_depth_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'friction_coefficient_depth_min.parquet')


    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()

