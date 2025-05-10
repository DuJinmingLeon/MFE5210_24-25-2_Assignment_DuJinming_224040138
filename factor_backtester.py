import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, ttest_1samp
import os
matplotlib.use('TkAgg')

class MinFactorBacktester:
    def __init__(self, symbol, freq, price_data, factor):
        """

        :param symbol: 期货代码
        :param freq: 回测频率
        :param price_data: 价格数据
        :param factor: 因子数据
        """
        self.symbol = symbol
        self.price_data = price_data
        self.freq = freq
        self.factor = factor
        self.merged_df = None
        self.nav = None
        self.statistics = {}
        self.factor_name = factor.columns[0]

    def merge_df(self, n):
        """
        计算因子n期后的收益率，并与因子数据合并
        """
        # 计算因子对应其n期后的收益率
        # mid_price = (self.price_data['high'] + self.price_data['low']) / 2
        self.price_data[f"return_{n}"] = ((self.price_data['open'].shift(-(n + 1)) - self.price_data['open'].shift(-1))
                                          / self.price_data['open'].shift(-1))
        # factor_return = (mid_price.shift(-(n + 1)) - mid_price.shift(-1)) / mid_price.shift(-1)
        # print(self.price_data)

        # 将factor和factor_return合并成一个DataFrame
        merged_df = pd.concat([self.price_data, self.factor], axis=1)
        merged_df = merged_df[[f'return_{n}', self.factor.columns[0]]]

        # 删除包含NaN的行
        merged_df.dropna(inplace=True)
        self.merged_df = merged_df
        return self.merged_df

    def calculate_correlation(self, period, spearman=True):
        """
        计算每period个数据的相关系数，计算IC和IR值
        返回IC和IR和p值
        """
        correlations = []
        if spearman:
            # 每隔period个数据计算一次相关系数
            for start in range(0, len(self.merged_df) - period + 1, period):
                end = start + period
                sliced_df = self.merged_df.iloc[start:end]

                # 计算该period段的Spearman相关系数
                correlation, _ = spearmanr(sliced_df.iloc[:, 0], sliced_df.iloc[:, 1])
                correlations.append(correlation)
        else:
            for start in range(0, len(self.merged_df) - period + 1, period):
                end = start + period
                sliced_df = self.merged_df.iloc[start:end]

                # 计算该period段的Spearman相关系数
                correlation, _ = pearsonr(sliced_df.iloc[:, 0], sliced_df.iloc[:, 1])
                correlations.append(correlation)

        # 计算IC和IR
        correlations = np.array(correlations)
        correlations = correlations[~np.isnan(correlations)]
        print(correlations)
        ic = np.mean(correlations)
        ir = ic / np.std(correlations)
        t_stat, p_value = ttest_1samp(correlations, 0)
        self.statistics = {'IC': ic, 'IR': ir, 'p_value': p_value}
        return ic, ir, p_value

    def calculate_nav(self, commission=False, commission_rate=0.0001):
        """
        计算策略净值
        """
        print(self.merged_df)
        self.merged_df['ret'] = self.merged_df[self.merged_df.columns[0]] * self.merged_df[self.merged_df.columns[1]]#self.merged_df[self.factor.columns[0]]

        self.merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.merged_df = self.merged_df.fillna(0)
        #print(self.merged_df)
        #self.merged_df.to_csv('E:\\arb_nav_test_1.csv')

        if commission:
            change = self.merged_df[self.factor.columns[0]] - self.merged_df[self.factor.columns[0]].shift(1,
                                                                                                           fill_value=0)

            #print(self.merged_df['ret'] - change.abs().fillna(0) * commission_rate)
            self.merged_df['actual_ret'] = self.merged_df['ret'] - change.abs().fillna(0) * commission_rate

            self.nav = (self.merged_df['ret'] - change.abs().fillna(0) * commission_rate).cumsum()
            #self.merged_df['actual_nav'] = self.nav
            #self.merged_df.to_csv('E:\\one_side_full_nav.csv')
            self.nav.name = 'nav'
        else:

            self.nav = self.merged_df['ret'].cumsum()
            self.nav.name = 'nav'
        self.merged_df['nav'] = self.nav

        #print(self.merged_df)
        #self.nav = self.nav.resample('1D', label='right').last().fillna(method='ffill')
        return self.nav


    def plot_nav(self):
        """
        画出策略净值曲线
        """
        #CAGR = (1+self.nav[-1]) ** (365 / len(self.nav)) - 1
        # 计算每日收益率
        daily_returns = self.nav.pct_change().dropna()
        # 计算年化收益率（几何平均）
        total_days = len(daily_returns)
        cagr = (self.nav[-1] / self.nav[0]) ** (252 / total_days) - 1 if total_days > 0 else 0
        # 计算年化波动率
        annualized_vol = daily_returns.std() * np.sqrt(252)
        # 计算夏普率（假设无风险利率为0）
        sharpe = cagr / annualized_vol if annualized_vol != 0 else 0

        self.statistics['sharpe'] = round(sharpe, 2)
        plt.plot(self.nav)
        plt.xlabel('Date')
        plt.ylabel('NAV')
        plt.title(f"{self.symbol}_{self.merged_df.columns[1]}_{self.freq}_{round(sharpe,2)}")
        plt.show()

    def save_output(self, path):
        """
        保存回测结果
        :param path:
        :return:
        """
        os.makedirs(path+f"/{self.factor_name}", exist_ok=True)
        # 计算每日收益率
        daily_returns = self.nav.pct_change().dropna()
        total_days = len(daily_returns)
        # 年化收益率（几何平均）
        cagr = (self.nav[-1] / self.nav[0]) ** (252 / total_days) - 1 if total_days > 0 else 0
        # 年化波动
        annualized_vol = daily_returns.std() * np.sqrt(252)
        # 夏普率
        sharpe = cagr / annualized_vol if annualized_vol != 0 else 0
        self.statistics['sharpe'] = round(sharpe, 2)
        self.nav.to_excel(path+f"/{self.factor_name}/{self.freq}_{self.symbol}_{round(sharpe,1)}.xlsx", index=True)

    @staticmethod
    def nav_corr(file_paths):
        """
        计算净值相关系数
        :param file_paths:
        :return:
        """
        # 用于存储每个Excel文件的DataFrame的列表
        dfs = []
        factor_name = []
        # 遍历所有的文件路径
        for file_path in file_paths:
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 将第一列设为索引
            df.set_index(df.columns[0], inplace=True)
            df.iloc[:, 0] = df.iloc[:, 0].diff()
            df.dropna(inplace=True)
            # 将DataFrame添加到列表
            dfs.append(df)
            factor_name.append(file_path.split('/')[-2])

        # 将所有DataFrame按列拼接
        combined_df = pd.concat(dfs, axis=1)

        # 计算相关系数矩阵
        correlation_matrix = combined_df.corr()
        correlation_matrix.columns = factor_name
        correlation_matrix.index = factor_name

        return correlation_matrix


class TickFactorBacktester:
    def __init__(self, symbol, freq, price_data, factor):
        """
        :param symbol: 期货代码
        :param freq: 回测频率
        :param price_data: 价格数据
        :param factor: 因子数据
        """
        self.symbol = symbol
        self.price_data = price_data
        self.freq = freq
        self.factor = factor
        self.merged_df = None
        self.nav = None
        self.statistics = {}
        self.factor_name = factor.columns[0]

    def merge_df(self, n):
        """
        计算因子n期后的收益率，并与因子数据合并
        """
        # 计算因子对应其n期后的收益率
        # mid_price = (self.price_data['a1'] + self.price_data['b1']) / 2
        if 'last' in self.price_data.columns:
            self.price_data[f'return_{n}'] = ((self.price_data['last'].shift(-(n + 1)) - self.price_data['last'].shift(-1)) /
                                              self.price_data['last'].shift(-1))
        elif 'open' in self.price_data.columns:
            self.price_data[f'return_{n}'] = ((self.price_data['open'].shift(-(n + 1)) - self.price_data['open'].shift(-1)) /
                                              self.price_data['open'].shift(-1))
        # print(self.price_data)
        # factor_return = (mid_price.shift(-(n + 1)) - mid_price.shift(-1)) / mid_price.shift(-1)

        # 将factor_return转换为DataFrame
        # factor_return = factor_return.to_frame(name=f'return_{n}')

        # 将factor和factor_return合并成一个DataFrame
        merged_df = pd.concat([self.price_data, self.factor], axis=1)

        # 删去使用了换月数据的行
        merged_df.reset_index(inplace=True)
        indices_delete = merged_df[merged_df['change'] == 'first'].index
        for idx in indices_delete:
            merged_df = merged_df.drop(range(idx, min(idx+n, len(merged_df))))

        merged_df.set_index('datetime', inplace=True)
        merged_df = merged_df[[f'return_{n}', self.factor.columns[0]]]

        # 删除包含NaN的行
        merged_df.dropna(inplace=True)
        self.merged_df = merged_df
        return self.merged_df

    def calculate_correlation(self, period, spearman=True):
        """
        计算每period个数据的相关系数，计算IC和IR值
        返回IC和IR和p值
        """
        correlations = []
        if spearman:
            # 每隔period个数据计算一次相关系数
            for start in range(0, len(self.merged_df) - period + 1, period):
                end = start + period
                sliced_df = self.merged_df.iloc[start:end]

                # 计算该period段的Spearman相关系数
                correlation, _ = spearmanr(sliced_df.iloc[:, 0], sliced_df.iloc[:, 1])
                correlations.append(correlation)
        else:
            for start in range(0, len(self.merged_df) - period + 1, period):
                end = start + period
                sliced_df = self.merged_df.iloc[start:end]

                # 计算该period段的Spearman相关系数
                correlation, _ = pearsonr(sliced_df.iloc[:, 0], sliced_df.iloc[:, 1])
                correlations.append(correlation)

        # 计算IC和IR
        correlations = np.array(correlations)
        correlations = correlations[~np.isnan(correlations)]
        ic = np.mean(correlations)
        ir = ic / np.std(correlations)
        t_stat, p_value = ttest_1samp(correlations, 0)
        self.statistics = {'IC': ic, 'IR': ir, 'p_value': p_value}
        return ic, ir, p_value

    def calculate_nav(self, commission=False, commission_rate=0.0001):
        """
        计算策略净值
        """
        self.merged_df['ret'] = self.merged_df[self.merged_df.columns[0]] * self.merged_df[self.factor.columns[0]]
        if commission:
            change = self.merged_df[self.factor.columns[0]] - self.merged_df[self.factor.columns[0]].shift(1,
                                                                                                           fill_value=0)
            self.nav = (self.merged_df['ret'] - change.abs().fillna(0) * commission_rate).cumsum()
            self.nav.name = 'nav'
        else:
            self.nav = self.merged_df['ret'].cumsum()
            self.nav.name = 'nav'
        self.merged_df['nav'] = self.nav

        self.nav = self.nav.resample('1D', label='right').last().fillna(method='ffill')
        return self.nav


    def plot_nav(self):
        """
        画出策略净值曲线
        """
        CAGR = (1+self.nav[-1]) ** (365 / len(self.nav)) - 1
        vol = self.nav.diff().std() * np.sqrt(250)
        sharpe = self.nav.diff().mean() * 250 /vol if vol != 0 else 0
        self.statistics['sharpe'] = round(sharpe, 1)
        plt.plot(self.nav)
        plt.xlabel('Date')
        plt.ylabel('NAV')
        plt.title(f"{self.symbol}_{self.merged_df.columns[1]}_{self.freq}_{round(sharpe,1)}")
        plt.show()

    def save_output(self, path, year):
        os.makedirs(path+f"/{self.factor_name}", exist_ok=True)
        CAGR = (1+self.nav[-1]) ** (365 / len(self.nav)) - 1
        vol = self.nav.diff().std() * np.sqrt(250)
        sharpe = self.nav.diff().mean() * 250 /vol if vol != 0 else 0
        self.statistics['sharpe'] = round(sharpe, 1)
        self.nav.to_excel(path+f"/{self.factor_name}/{self.freq}_{self.symbol}_{year}_{round(sharpe,1)}.xlsx", index=True)

    @staticmethod
    def nav_corr(file_paths):
        # 用于存储每个Excel文件的DataFrame的列表
        dfs = []
        factor_name = []
        # 遍历所有的文件路径
        for file_path in file_paths:
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 将第一列设为索引
            df.set_index(df.columns[0], inplace=True)
            df.iloc[:, 0] = df.iloc[:, 0].diff()
            df.dropna(inplace=True)
            # 将DataFrame添加到列表
            dfs.append(df)
            factor_name.append(file_path.split('/')[-2])

        # 将所有DataFrame按列拼接
        combined_df = pd.concat(dfs, axis=1)

        # 计算相关系数矩阵
        correlation_matrix = combined_df.corr()
        correlation_matrix.columns = factor_name
        correlation_matrix.index = factor_name

        return correlation_matrix



