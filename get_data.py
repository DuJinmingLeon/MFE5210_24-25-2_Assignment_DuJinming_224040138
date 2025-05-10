import h5py
import os
import numpy as np
import pandas as pd
import hdf5plugin
#import rqdatac as rqd
import sys
from multiprocessing import Pool


def read_data_for_date(date_str, full_file):
    """读取单个日期的数据， 主要用于并行读取"""
    with h5py.File(full_file, "r") as data:
        if date_str in data:
            # 读取该日期的数据并转换为 DataFrame
            data_df = pd.DataFrame(data[date_str][()])
            return data_df
        else:
            # print(f"No data found for {date_str}")
            return pd.DataFrame()

# 主要函数
# min数据时间区间2010-01-04 -- 2024-10-25；['open', 'high', 'low', 'close', 'open_interest', 'volume','total_turnover', 'date', 'open_interest_diff', 'ret']

# tick数据
#Index(['trading_date', 'open', 'last', 'high', 'low', 'prev_settlement',
#       'prev_close', 'volume', 'open_interest', 'total_turnover', 'limit_up',
#       'limit_down', 'a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4',
#       'b5', 'a1_v', 'a2_v', 'a3_v', 'a4_v', 'a5_v', 'b1_v', 'b2_v', 'b3_v',
#       'b4_v', 'b5_v', 'change_rate'],
def get_rdk_data(order_book_id, start_date, end_date, folder_path='E:\\Internship\\galaxy_commodity\\data',
                 frequency='min'):
    """
    :param folder_path: 米筐数据文件路径
    :param end_date: tick级数据需要 '2024-10-21'的字符串格式即可
    :param start_date: tick级数据需要 '2024-10-21'的字符串格式即可
    :param order_book_id: SC2308 字符串格式
    :param frequency: 'min' or 'tick' 字符串格式
    :return: pd.dataframe
    """
    os.environ['HDF5_PLUGIN_PATH'] = hdf5plugin.PLUGINS_PATH
    if frequency == 'min':
        full_file = os.path.join(folder_path, 'min', order_book_id + '.h5')
        data = h5py.File(full_file, "r")
        merged_df = pd.DataFrame(data['data'][()])
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], format="%Y%m%d%H%M%S")
        merged_df['date'] = pd.to_datetime(merged_df['datetime'].dt.date)
        merged_df['total_turnover'] = merged_df['total_turnover'] / 1000
        merged_df.loc[merged_df['datetime'].dt.time > pd.to_datetime('15:00:00').time(), 'date'] += pd.Timedelta(days=1)
        # 计算每个tick的open_interest
        merged_df['open_interest_diff'] = merged_df['open_interest'].diff()
        # 设置索引
        merged_df.set_index('datetime', inplace=True)
        # 删去第一行数据
        merged_df['ret'] = merged_df['close'].pct_change()
        merged_df = merged_df.drop(merged_df.index[0])


    elif frequency == 'tick':
        full_file = os.path.join(folder_path, 'ticks', order_book_id + '.h5')
        # data = h5py.File(full_file, "r")
        # 生成所有日期的范围
        date_range = pd.date_range(start=start_date, end=end_date)
        # 初始化一个空的 DataFrame
        #merged_df = pd.DataFrame()
        # 将日期转换为字符串格式的列表
        date_strs = [date.strftime('%Y%m%d') for date in date_range]

        # 使用多进程池来并行读取数据
        with Pool(processes=os.cpu_count()) as pool:  # 使用 CPU 的核心数
            results = pool.starmap(read_data_for_date, [(date_str, full_file) for date_str in date_strs])

        # 将所有结果合并到一个 DataFrame 中
        merged_df = pd.concat(results, ignore_index=True)

        # # 循环遍历每个日期
        # for date in date_range:
        #     date_str = date.strftime('%Y%m%d')  # 转换日期格式为字符串，如 '20230829'
        #
        #     # 检查 HDF5 文件中是否有该日期的数据
        #     if date_str in data:
        #         # 读取该日期的数据并转换为 DataFrame
        #         data_df = pd.DataFrame(data[date_str][()])
        #
        #         # 将数据追加到合并的 DataFrame 中
        #         merged_df = pd.concat([merged_df, data_df], ignore_index=True)

        # 关闭 HDF5 文件
        # data.close()
        # data_df = pd.DataFrame(data[date][()])
        merged_df['time'] = merged_df['time'].apply(lambda x: str(x).zfill(9))
        merged_df['date'] = merged_df['date'].astype(str)
        merged_df['datetime'] = pd.to_datetime(merged_df['date'] + merged_df['time'], format="%Y%m%d%H%M%S%f")
        merged_df[['last', 'high', 'low']] = merged_df[['last', 'high', 'low']] / 10000
        merged_df[['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5']] = merged_df[
                                                                                      ['a1', 'a2', 'a3', 'a4', 'a5',
                                                                                       'b1', 'b2', 'b3', 'b4',
                                                                                       'b5']] / 10000
        merged_df['total_turnover'] = merged_df['total_turnover'] / 1000
        # 删去a1=0的行
        merged_df = merged_df[(merged_df['a1'] != 0) | (merged_df['b1'] != 0)]
        merged_df['date'] = pd.to_datetime(merged_df['date'], format="%Y%m%d")
        # 对于交易时间大于15:00的数据，将其日期加一天
        merged_df.loc[merged_df['datetime'].dt.time > pd.to_datetime('15:00:00').time(), 'date'] += pd.Timedelta(days=1)
        # 删去不在交易时间的垃圾数据
        merged_df = merged_df[~((merged_df['datetime'].dt.time > pd.to_datetime('15:00:00').time()) &
                              (merged_df['datetime'].dt.time <= pd.to_datetime('20:58:50').time()))]
        # 计算每个tick的volume
        merged_df['volume_diff'] = merged_df.groupby('date')['volume'].diff()
        merged_df['volume_diff'].fillna(merged_df['volume'], inplace=True)
        # 计算每个tick的open_interest
        merged_df['open_interest_diff'] = merged_df['open_interest'].diff()
        # 处理换月
        change = pd.read_csv(folder_path + '\\month_rolling.csv')
        change['date'] = pd.to_datetime(change['date'])
        merged_df = pd.merge(merged_df, change[change['symbol'] == order_book_id[:2]], on='date', how='left').drop(
            columns='symbol')
        merged_df['change'] = merged_df['change'].fillna(method='bfill')
        merged_df['dominant'] = merged_df['dominant'].fillna(method='bfill')
        # 标记换月后的第一个数据
        first_indices = merged_df[merged_df['change'] == True].groupby('dominant').apply(lambda x: x.index[0])
        if len(first_indices) != 0:
            merged_df.loc[first_indices, 'change'] = 'first'


        # 设置索引
        merged_df.set_index('datetime', inplace=True)
        # 删去第一行数据
        merged_df = merged_df.drop(merged_df.index[0])

        # 获取 'volume' 和 'open_interest' 列的索引位置
        volume_index = merged_df.columns.get_loc('volume')
        open_interest_index = merged_df.columns.get_loc('open_interest')

        # 将 'volume_diff' 列插入到 'volume' 列之后
        merged_df.insert(volume_index + 1, 'volume_diff', merged_df.pop('volume_diff'))

        # 将 'open_interest_diff' 列插入到 'open_interest' 列之后
        merged_df.insert(open_interest_index + 2, 'open_interest_diff', merged_df.pop('open_interest_diff'))

        # merged_df = merged_df[merged_df['change'] == False]
    return merged_df


def tick_data_resample(data, freq):
    """
    将数据按照freq重新采样
    :param data: pd.DataFrame
    :param freq: str, 采样频率，如'1min', '5min', '1H', '1D'
    :return: pd.DataFrame
    """
    if freq != 'tick':
        # 重新采样
        data_resampled = data.resample(freq, closed='right',label='right').agg({'last': 'ohlc', 'dominant': 'last', 'change': 'last',
                                                                 'volume': 'last', 'volume_diff': 'sum',
                                                                 'open_interest': 'last', 'open_interest_diff': 'sum'})
        data_resampled.columns = data_resampled.columns.droplevel(0)
        data_resampled.columns.name = None
        # 去掉包含NaN的行
        data_resampled.dropna(inplace=True)
        return data_resampled
    else:
        return data

def min_data_resample(data, freq):
    """
    将数据按照freq重新采样
    :param data: pd.DataFrame
    :param freq: str, 采样频率，如'1min', '5min', '1H', '1D'
    :return: pd.DataFrame
    """
    if freq != '1min':
        # 重新采样
        data_resampled = data.resample(freq, closed='right', label='right').agg({ 'high': 'max', 'low': 'min', 'open': 'first',
                                                                  'close': 'last', 'volume': 'sum',
                                                                  'total_turnover': 'sum', 'open_interest': 'last',
                                                                  'open_interest_diff': 'sum'})
        data_resampled.columns.name = None
        data_resampled['ret'] = data_resampled['close'].pct_change()
        # 去掉包含NaN的行
        data_resampled.dropna(inplace=True)
        return data_resampled
    else:
        return data


def read_parquet_tick_data(contract_id,folder_path='E:\\Internship\\galaxy_commodity\\data'):
    data_path = folder_path + '\\ticks\\' + contract_id + '.parquet'
    tick_df = pd.read_parquet(data_path)
    tick_df = tick_df.reset_index()
    tick_df['trading_date'] = pd.to_datetime(tick_df['trading_date'])
    tick_df['date'] = tick_df['trading_date'].dt.date
    tick_df['date'] = pd.to_datetime(tick_df['date'])

    # 处理换月
    change = pd.read_csv(folder_path + '\\month_rolling.csv')
    change['date'] = pd.to_datetime(change['date'])

    merged_df = pd.merge(tick_df, change[change['symbol'] == contract_id[:2]], on='date', how='left').drop(
        columns='symbol')
    merged_df['change'] = merged_df['change'].fillna(method='bfill')
    merged_df['dominant'] = merged_df['dominant'].fillna(method='bfill')
    # 标记换月后的第一个数据
    first_indices = merged_df[merged_df['change'] == True].groupby('dominant').apply(lambda x: x.index[0])
    if len(first_indices) != 0:
        merged_df.loc[first_indices, 'change'] = 'first'
    # 设置索引
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
    merged_df.set_index('datetime', inplace=True)
    return merged_df


def min_month_rolling_clear(data_min,ticker,start_date,end_date,n):
    data = pd.read_csv('E:\\Internship\\galaxy_commodity\\data\\month_rolling.csv')
    data = data[data['symbol'] == ticker]
    data['date'] = pd.to_datetime(data['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = data[(data['date'] > start_date) & (data['date'] < end_date)]
    data = data[data['change'] == True]
    data = data.reset_index(drop=True)

    data_min = data_min[(data_min['date'] > start_date) & (data_min['date'] < end_date)]

    for index, row in data.iterrows():
        changing_date = row['date']
        target_index = data_min[data_min['date'] == changing_date].index[0]
        selected_rows = data_min.loc[target_index - n:target_index + n]
        print(selected_rows[['datetime','date','open','close']])
        indices_to_drop = [target_index + i for i in range(-n, n + 1)]
        data_min = data_min.drop(indices_to_drop)

    data_min = data_min.reset_index(drop=True)
    data_min = data_min.set_index('datetime')
    return data_min


def min_month_rolling_clear_easy(data_min,ticker,n):
    data = pd.read_csv('E:\\Internship\\galaxy_commodity\\data\\month_rolling.csv')
    data = data[data['symbol'] == ticker]
    data['date'] = pd.to_datetime(data['date'])

    #start_date = data_min.loc[data_min.index[0], 'date']
    #end_date = data_min.loc[data_min.index[-1], 'date']
    if 'date' not in list(data_min.columns):
        data_min['date'] = data_min.index.date

    start_date = data_min.first_valid_index()
    end_date = data_min.last_valid_index()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data = data[(data['date'] > start_date) & (data['date'] < end_date)]
    data = data[data['change'] == True]
    data = data.reset_index(drop=True)
    print(data)
    data_min = data_min.reset_index()
    for index, row in data.iterrows():
        changing_date = row['date']
        test = data_min[data_min['date'] == changing_date]
        if not (test.empty):
            target_index = list(data_min[data_min['date'] == changing_date].index)[0]
            selected_rows = data_min.loc[target_index - n:target_index + n]
            #print(selected_rows[['datetime','date','open','close']])
            indices_to_drop = [target_index + i for i in range(-n, n + 1)]
            data_min = data_min.drop(indices_to_drop)

    data_min = data_min.reset_index(drop=True)
    data_min = data_min.set_index('datetime')

    return data_min
