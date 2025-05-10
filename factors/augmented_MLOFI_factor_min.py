import pandas as pd
import numpy as np
from get_data import get_rdk_data
import warnings
from factor_backtester import MinFactorBacktester
warnings.filterwarnings('ignore')


def calculate_AMLOFI(flag, df_tick, n):
    if flag == 'Buy':
        # 取值情况 1
        condition1 = (df_tick['b' + str(n + 1)].shift(1) < df_tick['b5'])
        result1 = np.where(condition1, df_tick[[f'b{j + 1}_v' for j in range(n, 5)]].sum(axis=1), np.nan)

        result2 = np.nan
        for k in range(n + 2, 5):
            # 取值情况 2
            condition2 = (df_tick['b' + str(n + 1)] > df_tick['b' + str(n + 1)].shift(1)) & \
                         (df_tick['b' + str(n + 1)].shift(1) <= df_tick['b' + str(k)]) & \
                         (df_tick['b' + str(n + 1)].shift(1) > df_tick['b' + str(k + 1)])
            result2 = np.where(condition2,
                               df_tick[[f'b{j + 1}_v' for j in range(n, k)]].sum(axis=1) - df_tick[
                                   'b' + str(n + 1) + '_v'].shift(1), np.nan)

        result4 = np.nan
        for i in range(n + 2, 5):
            # 取值情况 4
            condition4 = (df_tick['b' + str(n + 1)] < df_tick['b' + str(n + 1)].shift(1)) & \
                         (df_tick['b' + str(n + 1)] <= df_tick['b' + str(i)].shift(1)) & \
                         (df_tick['b' + str(n + 1)] > df_tick['b' + str(i + 1)].shift(1))
            result4 = np.where(condition4, df_tick['b' + str(n + 1) + '_v'] - df_tick[
                [f'b{j + 1}_v' for j in range(n, i)]].shift(1).sum(axis=1), np.nan)

        # 取值情况 3
        condition3 = (df_tick['b' + str(n + 1)] == df_tick['b' + str(n + 1)].shift(1))
        result3 = np.where(condition3, df_tick['b' + str(n + 1) + '_v'] - df_tick['b' + str(n + 1) + '_v'].shift(1),
                           np.nan)

        # 取值情况 5
        condition5 = (df_tick['b' + str(n + 1)] < df_tick['b5'].shift(1))
        result5 = np.where(condition5,
                           -1 * df_tick[[f'b{j + 1}_v' for j in range(n, 5)]].shift(1).sum(axis=1),
                           np.nan)

    if flag == 'Sell':
        # 取值情况 1
        condition1 = (df_tick['a' + str(n + 1)] > df_tick['a5'].shift(1))
        result1 = np.where(condition1, df_tick[[f'a{j + 1}_v' for j in range(n, 5)]].shift(1).sum(axis=1),
                           np.nan)
        result2 = np.nan
        for k in range(n + 2, 5):
            # 取值情况 2
            condition2 = (df_tick['a' + str(n + 1)] > df_tick['a' + str(n + 1)].shift(1)) & \
                         (df_tick['a' + str(n + 1)] >= df_tick['a' + str(k)].shift(1)) & \
                         (df_tick['a' + str(n + 1)] < df_tick['a' + str(k + 1)].shift(1))
            result2 = np.where(condition2,
                               df_tick[[f'a{j + 1}_v' for j in range(n, k)]].shift(1).sum(axis=1) -
                               df_tick['a' + str(n + 1) + '_v'], np.nan)

        result4 = np.nan
        for i in range(n + 2, 5):
            # 取值情况 4
            condition4 = (df_tick['a' + str(n + 1)] < df_tick['a' + str(n + 1)].shift(1)) & \
                         (df_tick['a' + str(n + 1)].shift(1) >= df_tick['a' + str(i)]) & \
                         (df_tick['a' + str(n + 1)].shift(1) < df_tick['a' + str(i + 1)])
            result4 = np.where(condition4, df_tick['a' + str(n + 1) + '_v'].shift(1) - df_tick[
                [f'a{j + 1}_v' for j in range(n, i)]].sum(axis=1), np.nan)

        # 取值情况 3
        condition3 = (df_tick['a' + str(n + 1)] == df_tick['a' + str(n + 1)].shift(1))
        result3 = np.where(condition3, df_tick['a' + str(n + 1) + '_v'] - df_tick['a' + str(n + 1) + '_v'].shift(1),
                           np.nan)

        # 取值情况 5
        condition5 = (df_tick['a' + str(n + 1)].shift(1) > df_tick['a5'])
        result5 = np.where(condition5, -1 * df_tick[[f'a{j + 1}_v' for j in range(n, 5)]].sum(axis=1),
                           np.nan)

    result = np.where(condition1, result1, np.where(condition2, result2, np.where(condition3, result3,
                                                                                  np.where(condition4, result4,
                                                                                           result5))))
    return result


def AMLOFI_min(df_min, df_tick):
    for n in range(3):
        df_tick['delta_BidVolume' + str(n + 1)] = calculate_AMLOFI('Buy', df_tick, n)
        df_tick['delta_AskVolume' + str(n + 1)] = calculate_AMLOFI('Sell', df_tick, n)
    df_tick['AMLOFI'] = df_tick[[('delta_BidVolume' + str(j + 1)) for j in range(2)]].sum(axis=1) - df_tick[
        [('delta_AskVolume' + str(j + 1)) for j in range(2)]].sum(axis=1)
    df_tick['AMLOFI'] = df_tick['AMLOFI'].ffill()
    # sum for minute
    amlofi_sum = df_tick.groupby(['min_time', 'trading_date'])['AMLOFI'].sum().to_dict()
    df_min['AMLOFI_min'] = df_min.apply(lambda x: amlofi_sum.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['AMLOFI_min'] = df_min['AMLOFI_min'].ffill()

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
    #print(filtered_df.columns, tick_df.columns)

    # 因子计算
    filtered_df = AMLOFI_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['AMLOFI_min'])
    filtered_df = filtered_df.drop(columns='AMLOFI_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)

    factor_df = pd.DataFrame(merged_df['AMLOFI_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'AMLOFI_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")


    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()

