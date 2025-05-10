import pandas as pd
import numpy as np
from get_data import get_rdk_data
import warnings
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def kyle_lambda_min(df):
    df['delta_P'] = df['close'].diff()
    df['delta_P'] = df['delta_P'].fillna(0)
    df.loc[df['delta_P'] < 0, 'direction'] = -1
    df.loc[df['delta_P'] >= 0, 'direction'] = 1
    df['up'] = df['delta_P'].rolling(window=10).sum()
    df['down_sub'] = df['volume']*df['direction']
    df['down'] = df['down_sub'] .rolling(window=10).sum()
    df['kyle_lambda_min'] = df['up']/df['down']
    df['kyle_lambda_min'] = df['kyle_lambda_min'].ffill()

    return df['kyle_lambda_min'].values


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
    filtered_df['kyle_lambda_min'] = kyle_lambda_min(filtered_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['kyle_lambda_min'])
    filtered_df = filtered_df.drop(columns='kyle_lambda_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['kyle_lambda_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'kyle_lambda_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()








