import pandas as pd
import numpy as np
from get_data import get_rdk_data, read_parquet_tick_data
import warnings
import scipy.stats as stats
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


# 买方或卖方从低档位到高档位累加委托量
def calculate_cumulative_volume(df, direction):
    # 提取需要累加的列
    columns_to_cumsum = [direction + f'{i}_v' for i in range(1, 6)]
    cumu_vol_clolumns = [direction + f'CumuVolume{i}' for i in range(1, 6)]
    # 对需要累加的列进行累加操作
    cumsum_df = df[columns_to_cumsum].cumsum(axis=1)
    # 将累加后的结果合并回原始DataFrame
    df[columns_to_cumsum] = cumsum_df
    return df


# 买方或卖方从低档位到高档位累加委托量
def calculate_total_volume(df):
    df['BuyVolumeTotal'] = df['b1_v']+df['b2_v']+df['b3_v']+df['b4_v']+df['b5_v']
    df['SellVolumeTotal'] = df['a1_v']+df['a2_v']+df['a3_v']+df['a4_v']+df['a5_v']
    for i in range(1,6):
        df['b'+str(i)+'_v'] = df['b'+str(i)+'_v']/df['BuyVolumeTotal']
        df['a'+str(i)+'_v'] = df['a'+str(i)+'_v']/df['SellVolumeTotal']

    df['MP'] = (df['b1'] + df['a1'])/2
    return df


def cal_naer_slope(df,direction):
    df[direction+'down1'] = (df[direction+'1']/df['MP'] - 1).abs()
    for i in range(2,6):
        df[direction+'down'+str(i)] = (df[direction+str(i)]/df[direction+str(i-1)] - 1).abs()
        df[direction+'up'+str(i)] = df[direction+str(i)+'_v']-1 #/ df[direction + 'Volume0' + str(i - 1)] - 1

    result = 0
    for i in range(2, 6):
        up_col = direction+'up'+str(i)
        down_col = direction+'down'+str(i)
        result += df[up_col] / df[down_col]
    df[direction+'ratio_sum'] = result
    df[direction+'slope'] = 0.2*(df[direction+'1_v']/df[direction+'down1']+df[direction+'ratio_sum'])
    return df


def cal_orderbook_slope_naer_min(df_min, df_tick):
    df_tick = calculate_cumulative_volume(df_tick,'b')
    df_tick = calculate_cumulative_volume(df_tick,'a')
    df_tick = calculate_total_volume(df_tick)
    df_tick = cal_naer_slope(df_tick,'b')
    df_tick = cal_naer_slope(df_tick,'a')
    df_tick['fac_tick'] = df_tick['bslope'] - df_tick['aslope']

    # sum for minute
    slope_mean = df_tick.groupby(['min_time', 'trading_date'])['fac_tick'].mean().to_dict()
    df_min['slope_naer_min'] = df_min.apply(lambda x: slope_mean.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['slope_naer_min'] = df_min['slope_naer_min'].ffill()
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
    filtered_df = cal_orderbook_slope_naer_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['slope_naer_min'])
    factor_df['slope_naer_min'] = -1*factor_df['slope_naer_min']
    filtered_df = filtered_df.drop(columns='slope_naer_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['slope_naer_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'slope_naer_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()


