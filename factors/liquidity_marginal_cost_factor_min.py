import pandas as pd
import numpy as np
from get_data import get_rdk_data
import warnings
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
    df[cumu_vol_clolumns] = cumsum_df
    return df

# 计算平均成本
def calc_avg_cost(df, flag):
    for i in range(1, 6):
        df[flag + 'PriVol' + str(i)] = df[flag + str(i)] * df[flag + str(i) + '_v']
        df[flag + 'AvgCost' + str(i)] = df[[(flag + 'PriVol' + str(j + 1)) for j in range(i)]].sum(axis=1).div(
            df[flag + 'CumuVolume' + str(i)], axis=0)  # / df[flag+'CumuVolume' + str(i)]
    return df


def calculate_ols_slope(df):
    x_Bid_cols = [f'bCumuVolume{i}' for i in range(1, 6)]
    x_Ask_cols = [f'aCumuVolume{i}' for i in range(1, 6)]
    y_Bid_cols = [f'bAvgCost{i}' for i in range(1, 6)]
    y_Ask_cols = [f'aAvgCost{i}' for i in range(1, 6)]
    # 买方调整乘-1
    df[x_Bid_cols] = df[x_Bid_cols].apply(lambda x: x * -1)
    df[y_Bid_cols] = df[y_Bid_cols].apply(lambda x: x * -1)

    x_cols = x_Bid_cols + x_Ask_cols
    y_cols = y_Bid_cols + y_Ask_cols
    X = df[x_cols].values.reshape(-1, len(x_cols), 1)
    Y = df[y_cols].values.reshape(-1, len(y_cols), 1)
    X = np.concatenate((np.ones_like(X), X), axis=2)
    X_T = X.transpose(0, 2, 1)
    # 判断奇异矩阵
    determinant = np.linalg.det(X_T @ X)
    # 判断行列式是否为0
    if np.isclose(determinant, 0).any():
        print("The matrix is singular.")
        beta = np.linalg.pinv(X_T @ X) @ X_T @ Y
    else:
        beta = np.linalg.inv(X_T @ X) @ X_T @ Y

    slopes = beta[:, 1, 0]
    return pd.Series(slopes, index=df.index)


def slope_tick(df_tick):
    df_tick['Slope'] = np.nan
    df_tick = calculate_cumulative_volume(df_tick, 'b')
    df_tick = calculate_cumulative_volume(df_tick, 'a')
    df_tick = calc_avg_cost(df_tick, 'b')
    df_tick = calc_avg_cost(df_tick, 'a')
    df_tick['Slope'] = calculate_ols_slope(df_tick)
    df_tick['Slope'] = df_tick['Slope'].fillna(0)
    df_tick['fac_tick'] = df_tick['Slope'] / (((df_tick['b1'] + df_tick['a1']) / 2) ** 2)
    return df_tick


def LMC_min(df_min, df_tick):
    df_tick = slope_tick(df_tick)
    # sum for minute
    slope_mean = df_tick.groupby(['min_time', 'trading_date'])['fac_tick'].mean().to_dict()
    df_min['LMC_min'] = df_min.apply(lambda x: slope_mean.get((x['min_time'], x['trading_date']), 0), axis=1)
    df_min['LMC_min'] = df_min['LMC_min'].ffill()
    df_min['LMC_min'] = df_min['LMC_min'].rolling(window=10, min_periods=1).mean()
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
    filtered_df = LMC_min(filtered_df,tick_df)
    filtered_df = filtered_df.dropna()
    # 检查是否含有 inf 或 -inf
    has_inf = filtered_df.isin([np.inf, -np.inf]).any(axis=1)
    # 删除含有 inf 或 -inf 的行
    filtered_df = filtered_df[~has_inf]
    factor_df = pd.DataFrame(filtered_df['LMC_min'])
    filtered_df = filtered_df.drop(columns='LMC_min')
    # 初始化 MinFactorBacktester 实例
    backtester = MinFactorBacktester(symbol='AG88', freq='1D', price_data=filtered_df, factor=factor_df)
    merged_df = backtester.merge_df(n=1)
    print(merged_df)


    factor_df = pd.DataFrame(merged_df['LMC_min'])
    print(factor_df)
    factor_store_path = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    factor_df.to_parquet(factor_store_path + 'LMC_min.parquet')

    ic, ir, p_value = backtester.calculate_correlation(period=240*20, spearman=True)
    print("\nIC, IR, and p-value:")
    print(f"IC: {ic}")
    print(f"IR: {ir}")
    print(f"p-value: {p_value}")

    # 调用 calculate_nav 方法
    nav = backtester.calculate_nav(commission=False, commission_rate=0.000)
    backtester.plot_nav()


