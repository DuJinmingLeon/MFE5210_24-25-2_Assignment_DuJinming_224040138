import pandas as pd
import numpy as np
from get_data import get_rdk_data, read_parquet_tick_data
import warnings
import scipy.stats as stats
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def cal_gamma(df_tick):
    df_tick['S0'] = (df_tick['b1'] + df_tick['a1'])/2
    df_tick['gamma'] = df_tick['S0'] / (0.05**(1/3))
    return df_tick


def cal_pressure(df_tick):
    df_tick = cal_gamma(df_tick)
    shallow_book_lst = ['1','2']
    deep_book_lst = ['3', '4', '5']
    for level in shallow_book_lst:
        df_tick['BuyPressure' + level] = ((df_tick['b'+level] - df_tick['S0'])/(0.05**(2/3)))*(df_tick['gamma']/6 - 1/4 * (df_tick['b'+level]-df_tick['S0'])/(0.05**(1/3)) + (df_tick['gamma']/118)*((df_tick['b'+level]-df_tick['S0'])**2)/(0.05**(2/3)))
        df_tick['SellPressure' + level] = ((df_tick['a' + level]-df_tick['S0'])/(0.05 ** (2 / 3))) * (
                    df_tick['gamma']/6 - 1/4*(df_tick['a'+level]-df_tick['S0'])/(0.05**(1/3)) + (
                        df_tick['gamma']/118)*((df_tick['a'+level]-df_tick['S0'])**2)/(0.05**(2 / 3)))

    '''
    for dlevel in deep_book_lst:
        df_tick['BuyPressure'+dlevel] = (1/(df_tick['BuyPrice'+dlevel]-df_tick['S0']) + (df_tick['gamma']*(0.05**(1/3)))/((df_tick['BuyPrice'+dlevel]-df_tick['S0'])**2) + ((df_tick['gamma']*(0.05**(1/3)))**2)/((df_tick['BuyPrice'+dlevel]-df_tick['S0'])**3))/1e10
        df_tick['SellPressure' + dlevel] = (1/(df_tick['SellPrice'+dlevel]-df_tick['S0']) + (
                    df_tick['gamma']*(0.05**(1/3)))/((df_tick['SellPrice'+dlevel]-df_tick['S0'])**2) + (
                                                      (df_tick['gamma']*(0.05 ** (1 / 3)))**2)/(
                                                      (df_tick['SellPrice'+dlevel]-df_tick['S0'])**3))/1e10
    '''
    for i in range(1, len(shallow_book_lst)+1):
        df_tick['BuyVolPre'+str(i)] = df_tick['BuyPressure'+str(i)]*df_tick['b'+str(i)+'_v']
        df_tick['SellVolPre'+str(i)] = df_tick['SellPressure'+str(i)]*df_tick['a'+str(i)+'_v']

    df_tick['BuyPressure'] = (df_tick['BuyVolPre1']+df_tick['BuyVolPre2'])/(df_tick['b1_v']+df_tick['b2_v'])
    df_tick['SellPressure'] = (df_tick['SellVolPre1']+df_tick['SellVolPre2'])/(df_tick['a1_v']+df_tick['a2_v'])
    df_tick['PressureDiff'] = df_tick['BuyPressure'] - df_tick['SellPressure']

    #df_tick['PressureDiff'] = df_tick['BuyPressure01'] - df_tick['SellPressure01']
    return df_tick


def cal_order_flow_pressure_min(df_min, df_tick):
    df_tick = cal_pressure(df_tick)
    # sum for minute
    pressure_mean = df_tick.groupby(['min_time', 'trading_date'])['PressureDiff'].mean().to_dict()
    df_min['shallow_pressure_min'] = df_min.apply(lambda x: pressure_mean.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['shallow_pressure_min'] = df_min['shallow_pressure_min'].ffill()
    #df_min['shallow_pressure_min'] = -1*df_min['shallow_pressure_min']
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
    filtered_df = cal_order_flow_pressure_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['shallow_pressure_min'])
    filtered_df = filtered_df.drop(columns='shallow_pressure_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['shallow_pressure_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'shallow_pressure_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()



