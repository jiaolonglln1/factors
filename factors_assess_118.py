import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class FactorEvaluator:
    """
    因子评价类，用于处理因子数据、构建多空组合、计算评价指标和可视化结果

    参数:
    data_dir: str, 因子数据存放目录 (默认: 'stocks_data')
    output_dir: str, 结果输出目录 (默认: 'portfolio')
    group_num: int, 分组数量 (默认: 5)
    calc_ew: bool, 是否计算等权重组合 (默认: True)
    calc_vw: bool, 是否计算市值加权组合 (默认: True)
    """

    def __init__(self, data_dir='stocks_data', output_dir='portfolio',
                 group_num=5, calc_ew=True, calc_vw=True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.group_num = group_num
        self.calc_ew = calc_ew
        self.calc_vw = calc_vw

        # 初始化数据容器
        self.stock_data = {}
        self.char_dict = {}
        self.all_dates = []
        self.stocks = []
        self.char = []

        # 根据选择初始化组合DataFrame
        if self.calc_ew:
            self.ew_portfolio = None
            self.ew_portfolio_cumret = None
            self.factors_ew_assessment = None

        if self.calc_vw:
            self.vw_portfolio = None
            self.vw_portfolio_cumret = None
            self.factors_vw_assessment = None

    def load_data(self):
        """加载因子数据"""
        print(f"Loading data from {self.data_dir}...")
        stock_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        for file in stock_files:
            stock_code = [i for i in file.split('_') if i.isdigit()][0]
            df = pd.read_csv(os.path.join(self.data_dir, file), parse_dates=['date'])
            df.index = df['date']
            df.pop('date')
            self.stock_data[stock_code] = df

            # 获取所有因子名称（排除非因子列）
            char_list = [i for i in df.keys() if i not in ['asset', 'ret', 'xret', 'lag_me', 'log_me']]
            for c in char_list:
                self.char_dict.setdefault(c, 0)
                self.char_dict[c] += 1

        # 获取所有日期和股票代码
        self.all_dates = sorted(list(set().union(*[df.index for df in self.stock_data.values()])))
        self.stocks = list(self.stock_data.keys())
        self.char = list(self.char_dict.keys())

        # 根据选择初始化组合DataFrame
        if self.calc_ew:
            self.ew_portfolio = pd.DataFrame(index=self.all_dates, columns=self.char)

        if self.calc_vw:
            self.vw_portfolio = pd.DataFrame(index=self.all_dates, columns=self.char)

    def build_long_short_portfolios(self):
        """构建多空组合"""
        print(f"Building long-short portfolios with {self.group_num} groups...")

        for t in self.all_dates:
            tindex = self.all_dates.index(t)
            valid_stock = [s for s in self.stocks if t in self.stock_data[s].index]

            for c in self.char:
                # 获取当前日期的因子数据
                raw_data = {s: self.stock_data[s].loc[t, c] for s in valid_stock if c in self.stock_data[s].keys()}

                # 剔除因子样本点数量小于分组数的日期
                if len(raw_data.keys()) < self.group_num:
                    continue

                # 按因子值排序
                sorted_data = sorted(raw_data, key=lambda x: raw_data[x])
                groupsize = round(len(sorted_data) / self.group_num)

                # 构建空头和多头组合
                short_stock = [i for i in sorted_data[:groupsize - 1]]  # 最小1/group_num组
                long_stock = [i for i in sorted_data[-groupsize:-1]]  # 最大1/group_num组

                # 计算等权重组合收益
                if self.calc_ew:
                    ew_short_ret = np.mean([self.stock_data[i].loc[t, 'xret'] for i in short_stock])
                    ew_long_ret = np.mean([self.stock_data[i].loc[t, 'xret'] for i in long_stock])
                    ew_portfolio_ret = ew_long_ret - ew_short_ret
                    self.ew_portfolio.loc[t, c] = ew_portfolio_ret

                # 计算市值加权组合收益
                if self.calc_vw and tindex > 0:  # 从第二期开始使用前一期的市值作为权重
                    short_stock_size = [self.stock_data[i].loc[self.all_dates[tindex - 1], 'size']
                                        if self.all_dates[tindex - 1] in self.stock_data[i].index
                                        else 1 / len(short_stock) for i in short_stock]
                    long_stock_size = [self.stock_data[i].loc[self.all_dates[tindex - 1], 'size']
                                       if self.all_dates[tindex - 1] in self.stock_data[i].index
                                       else 1 / len(short_stock) for i in long_stock]

                    # 计算市值权重
                    short_stock_w = [w / np.sum(short_stock_size) for w in short_stock_size]
                    long_stock_w = [w / np.sum(long_stock_size) for w in long_stock_size]

                    # 计算市值权重组合收益
                    vw_short_ret = np.dot(short_stock_w, [self.stock_data[i].loc[t, 'xret'] for i in short_stock])
                    vw_long_ret = np.dot(long_stock_w, [self.stock_data[i].loc[t, 'xret'] for i in long_stock])
                    vw_portfolio_ret = vw_long_ret - vw_short_ret
                    self.vw_portfolio.loc[t, c] = vw_portfolio_ret

        # 按日期排序
        if self.calc_ew:
            self.ew_portfolio.sort_index(inplace=True)
        if self.calc_vw:
            self.vw_portfolio.sort_index(inplace=True)

    def calculate_cumulative_returns(self, plot=True):
        """计算累积收益率"""
        print("Calculating cumulative returns...")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 计算等权重组合累积收益率
        if self.calc_ew:
            self.ew_portfolio_cumret = pd.DataFrame()
            for c in self.char:
                if not self.ew_portfolio[c].isnull().all():
                    self.ew_portfolio_cumret[f'{c}_cumret'] = (1 + self.ew_portfolio[c]).cumprod()

                    # 绘制累积收益率图
                    if plot:
                        plt.figure()
                        plt.title(f'ew_{c}_cumret')
                        plt.plot(self.ew_portfolio_cumret.index, self.ew_portfolio_cumret[f'{c}_cumret'])
                        plot_dir = os.path.join(self.output_dir, f'plot/ew_cumret_{self.group_num}')
                        os.makedirs(plot_dir, exist_ok=True)
                        plt.savefig(f"{plot_dir}/ew_{c}.png", format="PNG", dpi=300)
                        plt.close()

        # 计算市值加权组合累积收益率
        if self.calc_vw:
            self.vw_portfolio_cumret = pd.DataFrame()
            for c in self.char:
                if not self.vw_portfolio[c].isnull().all():
                    self.vw_portfolio_cumret[f'{c}_cumret'] = (1 + self.vw_portfolio[c]).cumprod()

                    # 绘制累积收益率图
                    if plot:
                        plt.figure()
                        plt.title(f'vw_{c}_cumret')
                        plt.plot(self.vw_portfolio_cumret.index, self.vw_portfolio_cumret[f'{c}_cumret'])
                        plot_dir = os.path.join(self.output_dir, f'plot/vw_cumret_{self.group_num}')
                        os.makedirs(plot_dir, exist_ok=True)
                        plt.savefig(f"{plot_dir}/vw_{c}.png", format="PNG", dpi=300)
                        plt.close()

    def evaluate_factors(self):
        """计算各因子评价指标"""
        print("Evaluating factors...")

        # 初始化评价指标DataFrame
        if self.calc_ew:
            self.factors_ew_assessment = pd.DataFrame(index=self.char)

            # 计算平均收益率、年化收益率、波动率等
            for c in self.char:
                if not self.ew_portfolio[c].isnull().all():
                    returns = self.ew_portfolio[c]
                    self.factors_ew_assessment.loc[c, 'mean_return'] = returns.mean()
                    self.factors_ew_assessment.loc[c, 'annualized_return'] = (1 + returns.mean()) ** 12 - 1
                    self.factors_ew_assessment.loc[c, 'volatility'] = returns.std()
                    self.factors_ew_assessment.loc[c, 'sharpe_ratio'] = returns.mean() / returns.std()

        if self.calc_vw:
            self.factors_vw_assessment = pd.DataFrame(index=self.char)

            # 计算平均收益率、年化收益率、波动率等
            for c in self.char:
                if not self.vw_portfolio[c].isnull().all():
                    returns = self.vw_portfolio[c]
                    self.factors_vw_assessment.loc[c, 'mean_return'] = returns.mean()
                    self.factors_vw_assessment.loc[c, 'annualized_return'] = (1 + returns.mean()) ** 12 - 1
                    self.factors_vw_assessment.loc[c, 'volatility'] = returns.std()
                    self.factors_vw_assessment.loc[c, 'sharpe_ratio'] = returns.mean() / returns.std()

    def save_results(self):
        """保存结果到文件"""
        print(f"Saving results to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)

        if self.calc_ew:
            self.factors_ew_assessment.to_csv(os.path.join(self.output_dir, f'factors_ew_assessment_{self.group_num}.csv'))

        if self.calc_vw:
            self.factors_vw_assessment.to_csv(os.path.join(self.output_dir, f'factors_vw_assessment_{self.group_num}.csv'))

    def run_full_analysis(self, plot=True):
        """
        运行完整的因子分析流程

        参数:
        plot: bool, 是否绘制累积收益率图 (默认: True)
        """
        self.load_data()
        self.build_long_short_portfolios()
        self.calculate_cumulative_returns(plot=plot)
        self.evaluate_factors()
        self.save_results()

        print("Analysis completed!")

        # 返回结果字典
        results = {}
        if self.calc_ew:
            results.update({
                'ew_portfolio': self.ew_portfolio,
                'ew_portfolio_cumret': self.ew_portfolio_cumret,
                'factors_ew_assessment': self.factors_ew_assessment
            })

        if self.calc_vw:
            results.update({
                'vw_portfolio': self.vw_portfolio,
                'vw_portfolio_cumret': self.vw_portfolio_cumret,
                'factors_vw_assessment': self.factors_vw_assessment
            })

        return results


# 使用示例
if __name__ == "__main__":
    # 示例：同时计算两种组合
    evaluator_both = FactorEvaluator(
        data_dir='stocks_data',
        output_dir='portfolio',
        group_num=5,
        calc_ew=True,
        calc_vw=True
    )
    results_both = evaluator_both.run_full_analysis()