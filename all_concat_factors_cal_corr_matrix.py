import os
import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
#from factor_backtester import MinFactorBacktester
from get_data import min_month_rolling_clear_easy
import pyarrow as pa
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    path_to_read_parquet_files = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\AG\\'
    path_to_store_files = 'E:\\Schoolwork Files\\MScFE\\MFE5210_Algo_Trading\\Assignment\\data\\factor_data\\'
    parquet_files = [os.path.join(path_to_read_parquet_files, f) for f in os.listdir(path_to_read_parquet_files) if
                     f.endswith('.parquet')]
    # 初始化一个空的 DataFrame
    merged_df = pd.DataFrame()
    # 遍历每个 Parquet 文件，读取并合并到 merged_df
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        merged_df = pd.concat([merged_df, df],axis=1)  # , ignore_index=True)

    #merged_df = merged_df.sort_index()
    # 检查是否有重复索引
    if merged_df.index.duplicated().any():
        print("存在重复索引，正在删除重复项...")
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]  # 保留第一个重复项
    merged_df = merged_df.ffill()
    # 打印合并后的 DataFrame
    print(merged_df)
    print(merged_df.columns)

    merged_df.to_parquet(path_to_store_files + 'ALL_AG_min_factors.parquet')
    corr_matrix = merged_df.corr()
    print(corr_matrix)
    corr_matrix.to_csv(path_to_store_files + "correlation_matrix.csv", index=True)
