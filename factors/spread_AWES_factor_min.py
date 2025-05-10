import pandas as pd
import numpy as np
from get_data import get_rdk_data, read_parquet_tick_data
import warnings
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def cal_tick_spread_amount(df_tick):
    # 买一卖一中间价版本
    df_tick['mid_price'] = (df_tick['b1'] + df_tick['a1'])/2
    df_tick['spread_tick'] = df_tick['last'] / df_tick['mid_price']
    df_tick['spread_tick'] = df_tick['spread_tick'].apply(np.log)
    #df_tick['spread_tick'] = df_tick['spread_tick'].apply(lambda x: np.abs(x))
    df_tick['TradeAmount'] = df_tick['volume'] * df_tick['mid_price']
    df_tick['up'] = df_tick['spread_tick'] * df_tick['TradeAmount']
    #df_tick['up'] = df_tick['spread_tick'] * df_tick['total_turnover']
    return df_tick


def cal_spread_AWES_min(df_min,df_tick_M01):
    df_tick_M01 = cal_tick_spread_amount(df_tick_M01)
    fac_tick_mean_M01 = df_tick_M01.groupby(['min_time', 'trading_date'])['up'].sum().to_dict()
    df_min['spread_AWES_min'] = df_min.apply(lambda x: fac_tick_mean_M01.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['spread_AWES_min'] = df_min['spread_AWES_min']/df_min['total_turnover']
    df_min['spread_AWES_min'] = df_min['spread_AWES_min'].ffill()
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
    filtered_df = cal_spread_AWES_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['spread_AWES_min'])
    filtered_df = filtered_df.drop(columns='spread_AWES_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['spread_AWES_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'spread_AWES_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()

