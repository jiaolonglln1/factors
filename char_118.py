import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import datetime
from WindPy import w
import sys
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import os  # 引入os模块用于文件夹操作

# 模块级常量：默认数据字段
DEFAULT_DATA_FIELDS = ("stm_issuingdate,close,mkt_cap_ard,mkt_freeshares,ev1,amt,turn,"
                       "qfa_tot_oper_rev,qfa_oper_rev,qfa_tot_oper_cost,qfa_oper_cost,qfa_opprofit,qfa_net_profit_is,"
                       "qfa_selling_dist_exp,qfa_gerl_admin_exp,qfa_fin_exp_is,qfa_fin_int_exp,qfa_rd_exp,qfa_tax,"
                       "qfa_grossmargin,qfa_tot_profit,tot_assets,tot_cur_assets,monetary_cap,acct_rcv,inventories,"
                       "oth_cur_assets,oth_non_cur_assets,fix_assets,networkingcapital,wgsd_invest_trading,wgsd_cce,"
                       "tot_liab,tot_cur_liab,st_borrow,acct_payable,netdebt,tot_equity,qfa_net_cash_flows_oper_act,"
                       "other_equity_instruments_PRE")

# 模块级常量：默认columns_to_update
DEFAULT_COLUMNS_TO_UPDATE = [
    'QFA_TOT_OPER_REV', 'QFA_OPER_REV', 'QFA_TOT_OPER_COST', 'QFA_OPER_COST', 'QFA_OPPROFIT',
    'QFA_NET_PROFIT_IS', 'QFA_SELLING_DIST_EXP', 'QFA_GERL_ADMIN_EXP',
    'QFA_FIN_EXP_IS', 'QFA_FIN_INT_EXP', 'QFA_RD_EXP', 'QFA_TAX',
    'QFA_GROSSMARGIN', 'QFA_TOT_PROFIT', 'TOT_ASSETS', 'TOT_CUR_ASSETS',
    'MONETARY_CAP', 'ACCT_RCV', 'INVENTORIES', 'OTH_CUR_ASSETS',
    'OTH_NON_CUR_ASSETS', 'FIX_ASSETS', 'NETWORKINGCAPITAL',
    'WGSD_INVEST_TRADING', 'WGSD_CCE', 'TOT_LIAB', 'TOT_CUR_LIAB',
    'ST_BORROW', 'ACCT_PAYABLE', 'NETDEBT', 'TOT_EQUITY',
    'QFA_NET_CASH_FLOWS_OPER_ACT', 'OTHER_EQUITY_INSTRUMENTS_PRE'
]


class StockAnalyzer:
    def __init__(self, stock_codes=["600519.SH"], start_date="2010-01-01", end_date="2025-03-09",
                 data_fields=None, columns_to_update=None, output_dir="output"):
        """初始化StockAnalyzer类，支持多只股票代码和自定义字段"""
        self.stock_codes = stock_codes if isinstance(stock_codes, list) else [stock_codes]
        self.start_date = start_date
        self.end_date = end_date
        self.df_dict = {}
        self.df_rev_dict = {}
        self.df_char_dict = {}
        self.df_bond = None
        self.success_count = 0
        self.output_dir = output_dir  # 添加输出文件夹参数，默认为"output"

        self.data_fields = data_fields if data_fields is not None else DEFAULT_DATA_FIELDS
        self.columns_to_update = columns_to_update if columns_to_update is not None else DEFAULT_COLUMNS_TO_UPDATE

        self.connect_wind()

    def connect_wind(self):
        """连接到WindPy数据服务"""
        w.start()
        if not w.isconnected():
            raise ConnectionError("无法连接到WindPy")

    def update_fields(self, new_fields):
        """更新数据字段"""
        self.data_fields = new_fields
        print(f"已更新数据字段为: {self.data_fields}")

    def update_columns(self, new_columns):
        """更新columns_to_update"""
        self.columns_to_update = new_columns
        print(f"已更新columns_to_update为: {self.columns_to_update}")

    def fetch_data(self):
        """为每只股票获取数据，记录进度并处理错误"""
        for idx, stock_code in enumerate(tqdm(self.stock_codes, desc="Fetching Data")):
            try:
                print(f"正在获取第 {idx + 1} 只股票 {stock_code} 的数据...")
                data_stock = w.wsd(stock_code, self.data_fields, self.start_date, self.end_date,
                                   "unit=1;rptType=1;Days=Alldays;Currency=CNY;PriceAdj=F")

                if data_stock.ErrorCode != 0:
                    raise ValueError(f"获取 {stock_code} 数据失败，错误码: {data_stock.ErrorCode}")

                if not self.df_dict:
                    data_index = w.wsd("881001.WI", "close", self.start_date, self.end_date,
                                       "Days=Alldays;Currency=CNY;PriceAdj=F")
                    data_bond = w.edb("S0059742", self.start_date, self.end_date)

                    if data_index.ErrorCode != 0 or data_bond.ErrorCode != 0:
                        raise ValueError("获取指数或债券数据失败")

                    df_index = pd.DataFrame(data_index.Data).T
                    df_index.columns = data_index.Fields

                    self.df_bond = pd.DataFrame(data_bond.Data).T
                    self.df_bond.index = pd.to_datetime(data_bond.Times)
                    self.df_bond.columns = data_bond.Fields

                df = pd.DataFrame(data_stock.Data).T
                df.columns = data_stock.Fields
                df.insert(0, 'date', data_stock.Times)
                df.insert(3, 'INDEX_CLOSE', df_index['CLOSE'])
                self.df_dict[stock_code] = df
                self.success_count += 1
                print(f"成功获取 {stock_code} 的数据，已完成 {self.success_count}/{len(self.stock_codes)} 只股票")

            except Exception as e:
                print(f"错误：获取 {stock_code} 数据时失败，原因: {e}")
                print(f"程序终止，已成功获取 {self.success_count} 只股票的数据")
                self.save_to_csv()
                sys.exit(1)

    def preprocess_data(self):
        """为每只股票预处理数据"""
        for stock_code in tqdm(self.df_dict.keys(), desc="Preprocessing Data"):
            df = self.df_dict[stock_code]
            df_rev = df.copy()
            df_rev['date'] = pd.to_datetime(df_rev['date'])
            df_rev['STM_ISSUINGDATE'] = pd.to_datetime(df_rev['STM_ISSUINGDATE'])

            issuing_dates = df_rev[df_rev['STM_ISSUINGDATE'].notna()][['STM_ISSUINGDATE']].copy()
            issuing_dates['original_index'] = issuing_dates.index
            issuing_dates = issuing_dates.join(df[self.columns_to_update], on='original_index')
            conflicts = []

            for idx, row in issuing_dates.iterrows():
                issuing_date = row['STM_ISSUINGDATE']
                target_row = df_rev[df_rev['date'] == issuing_date]

                if not target_row.empty:
                    target_idx = target_row.index[0]
                    amt_value = target_row['AMT'].iloc[0]
                    target_issuing_date = target_row['STM_ISSUINGDATE'].iloc[0]

                    if pd.notna(target_issuing_date) and target_issuing_date != 'revised':
                        conflicts.append((target_idx, target_issuing_date))

                    if pd.notna(amt_value):
                        df_rev.at[target_idx, 'STM_ISSUINGDATE'] = 'revised'
                        for col in self.columns_to_update:
                            if col in df_rev.columns:
                                df_rev.at[target_idx, col] = df_rev.at[idx, col]
                    else:
                        subsequent_rows = df_rev[df_rev['date'] > issuing_date].sort_values(by='date')
                        for sub_idx, sub_row in subsequent_rows.iterrows():
                            if pd.notna(sub_row['AMT']):
                                sub_issuing_date = sub_row['STM_ISSUINGDATE']
                                if pd.notna(sub_issuing_date) and sub_issuing_date != 'revised':
                                    conflicts.append((sub_idx, sub_issuing_date))
                                df_rev.at[sub_idx, 'STM_ISSUINGDATE'] = 'revised'
                                for col in self.columns_to_update:
                                    if col in df_rev.columns:
                                        df_rev.at[sub_idx, col] = df_rev.at[idx, col]
                                break

                for col in self.columns_to_update:
                    if col in df_rev.columns:
                        df_rev.at[idx, col] = np.nan
                df_rev.at[idx, 'STM_ISSUINGDATE'] = np.nan

            for target_idx, original_issuing_date in conflicts:
                original_row = issuing_dates[issuing_dates['STM_ISSUINGDATE'] == original_issuing_date]
                if not original_row.empty:
                    original_idx = original_row['original_index'].iloc[0]
                    target_row = df_rev[df_rev['date'] == original_issuing_date]
                    if not target_row.empty:
                        new_target_idx = target_row.index[0]
                        amt_value = target_row['AMT'].iloc[0]
                        if pd.notna(amt_value):
                            df_rev.at[new_target_idx, 'STM_ISSUINGDATE'] = 'revised'
                            for col in self.columns_to_update:
                                if col in df_rev.columns:
                                    df_rev.at[new_target_idx, col] = issuing_dates.at[original_idx, col]
                        else:
                            subsequent_rows = df_rev[df_rev['date'] > original_issuing_date].sort_values(by='date')
                            for sub_idx, sub_row in subsequent_rows.iterrows():
                                if pd.notna(sub_row['AMT']):
                                    df_rev.at[sub_idx, 'STM_ISSUINGDATE'] = 'revised'
                                    for col in self.columns_to_update:
                                        if col in df_rev.columns:
                                            df_rev.at[sub_idx, col] = issuing_dates.at[original_idx, col]
                                    break

            df_rev['date'] = pd.to_datetime(df_rev['date'])
            df_rev.insert(1, 'year_month', df_rev['date'].dt.to_period('M'))
            df_rev.insert(2, 'year_quarter', df_rev['date'].dt.to_period('Q'))
            df_rev.insert(3, 'year', df_rev['date'].dt.to_period('Y'))
            df_rev[self.columns_to_update] = df_rev[self.columns_to_update].fillna(method='ffill')
            df_rev = df_rev.dropna(subset=['AMT', 'STM_ISSUINGDATE'], how='all')
            df_rev.index = df_rev['date']
            df_rev.insert(7, 'BOND_RATE', (1 + self.df_bond['CLOSE'] / 100) ** (1 / 12) - 1)
            df_rev.insert(5, 'RETURN', df_rev['CLOSE'].pct_change())
            self.df_rev_dict[stock_code] = df_rev

    def calculate_characteristics(self):
        """为每只股票计算特征"""
        for stock_code in tqdm(self.df_rev_dict.keys(), desc="Calculating Characteristics"):
            df_rev = self.df_rev_dict[stock_code]
            last_day = df_rev.groupby('year_month')['date'].idxmax()
            self.df_char_dict[stock_code] = pd.DataFrame(index=df_rev.loc[last_day]['date'])

            df_rev['AMT'] = pd.to_numeric(df_rev['AMT'])
            r_a = abs(df_rev['RETURN']) / df_rev['AMT']
            df_sup = df_rev.copy()
            df_sup['r_a'] = r_a

            df_rev['MKT_CAP_ARD'] = pd.to_numeric(df_rev['MKT_CAP_ARD'])
            df_rev['TURN'] = pd.to_numeric(df_rev['TURN'])
            self.df_char_dict[stock_code]['ret'] = df_rev['RETURN']
            self.df_char_dict[stock_code]['xret'] = df_rev['RETURN'] - df_rev['BOND_RATE']
            self.df_char_dict[stock_code]['lag_me'] = df_rev.loc[last_day]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['log_me'] = np.log(df_rev.loc[last_day]['MKT_CAP_ARD'])
            self.df_char_dict[stock_code]['size'] = np.log(df_rev.loc[last_day]['MKT_CAP_ARD'])
            self.df_char_dict[stock_code]['size3'] = np.power(self.df_char_dict[stock_code]['size'], 3)

            mon_avg = df_rev.groupby('year_month')['TURN'].mean()
            mon3_avg = mon_avg.shift(1).rolling(3).mean()
            mon6_avg = mon_avg.shift(1).rolling(6).mean()
            mon12_avg = mon_avg.shift(1).rolling(12).mean()
            mon6_std = mon_avg.shift(1).rolling(6).std()
            self.df_char_dict[stock_code]['turnm'] = self.df_char_dict[stock_code].index.to_period('M').map(
                np.log(mon6_avg))
            self.df_char_dict[stock_code]['turnq'] = self.df_char_dict[stock_code].index.to_period('M').map(
                np.log(mon3_avg))
            self.df_char_dict[stock_code]['turna'] = self.df_char_dict[stock_code].index.to_period('M').map(
                np.log(mon12_avg))
            self.df_char_dict[stock_code]['vturn'] = self.df_char_dict[stock_code].index.to_period('M').map(mon6_std)
            self.df_char_dict[stock_code]['cvturn'] = self.df_char_dict[stock_code].index.to_period('M').map(
                mon6_std / mon6_avg)
            self.df_char_dict[stock_code]['abturn'] = self.df_char_dict[stock_code].index.to_period('M').map(
                mon_avg / mon12_avg)

            dtvm_avg = df_rev.groupby('year_month')['AMT'].mean()
            dtvq_avg = dtvm_avg.shift(1).rolling(3).mean()
            dtva_avg = dtvm_avg.shift(1).rolling(12).mean()
            dtvm_std = dtvm_avg.shift(1).rolling(6).std()
            dtvb_avg = dtvm_avg.shift(1).rolling(6).mean()
            self.df_char_dict[stock_code]['dtvm'] = self.df_char_dict[stock_code].index.to_period('M').map(
                dtvm_avg.shift(1))
            self.df_char_dict[stock_code]['dtvq'] = self.df_char_dict[stock_code].index.to_period('M').map(dtvq_avg)
            self.df_char_dict[stock_code]['dtva'] = self.df_char_dict[stock_code].index.to_period('M').map(dtva_avg)
            self.df_char_dict[stock_code]['vdtv'] = self.df_char_dict[stock_code].index.to_period('M').map(dtvm_std)
            self.df_char_dict[stock_code]['cvd'] = self.df_char_dict[stock_code].index.to_period('M').map(
                dtvm_std / dtvb_avg)
            self.df_char_dict[stock_code]['ami'] = self.df_char_dict[stock_code].index.to_period('M').map(
                df_sup.groupby('year_month')['r_a'].mean().shift(1))
            self.df_char_dict[stock_code]['tv'] = self.df_char_dict[stock_code].index.to_period('M').map(
                df_rev.groupby('year_month')['RETURN'].std().shift(1))
            self.df_char_dict[stock_code]['ts'] = self.df_char_dict[stock_code].index.to_period('M').map(
                df_rev.groupby('year_month')['RETURN'].skew().shift(1))

            self.df_char_dict[stock_code]['m1'] = df_rev.loc[last_day]['CLOSE'].pct_change()
            self.df_char_dict[stock_code]['m11'] = (df_rev.loc[last_day]['CLOSE'].shift(1) - df_rev.loc[last_day][
                'CLOSE'].shift(12)) / df_rev.loc[last_day]['CLOSE'].shift(12)
            self.df_char_dict[stock_code]['m3'] = (df_rev.loc[last_day]['CLOSE'].shift(1) - df_rev.loc[last_day][
                'CLOSE'].shift(3)) / df_rev.loc[last_day]['CLOSE'].shift(3)
            self.df_char_dict[stock_code]['m6'] = (df_rev.loc[last_day]['CLOSE'].shift(1) - df_rev.loc[last_day][
                'CLOSE'].shift(6)) / df_rev.loc[last_day]['CLOSE'].shift(6)
            self.df_char_dict[stock_code]['m60'] = (df_rev.loc[last_day]['CLOSE'].shift(1) - df_rev.loc[last_day][
                'CLOSE'].shift(60)) / df_rev.loc[last_day]['CLOSE'].shift(60)
            self.df_char_dict[stock_code]['m24'] = (df_rev.loc[last_day]['CLOSE'].shift(12) - df_rev.loc[last_day][
                'CLOSE'].shift(35)) / df_rev.loc[last_day]['CLOSE'].shift(35)
            self.df_char_dict[stock_code]['mchg'] = self.df_char_dict[stock_code]['m6'] - (
                        (df_rev.loc[last_day]['CLOSE'].shift(7) - df_rev.loc[last_day]['CLOSE'].shift(12)) /
                        df_rev.loc[last_day]['CLOSE'].shift(12))

            monthly = df_rev['CLOSE'].resample('M').last()
            rolling_max = df_rev['CLOSE'].rolling(window=252).max()
            monthly_max = rolling_max.resample('M').last()
            ratio = monthly / monthly_max
            self.df_char_dict[stock_code]['52w'] = ratio.reindex(self.df_char_dict[stock_code].index, method='ffill')
            self.df_char_dict[stock_code]['mdr'] = (
                df_rev['RETURN'].rolling(22).apply(lambda x: x.nlargest(5).mean()).resample('M').last()).reindex(
                self.df_char_dict[stock_code].index, method='ffill')
            self.df_char_dict[stock_code]['mdr'].iloc[1] = 0.016860
            self.df_char_dict[stock_code]['pr'] = df_rev.loc[last_day]['CLOSE']

            daily_ret = pd.DataFrame(df_rev['RETURN'])  # 获取股票日收益率
            market_ret = pd.DataFrame((df_rev['INDEX_CLOSE'] - df_rev['INDEX_CLOSE'].shift(1)) / df_rev['INDEX_CLOSE'])  # 市场日收益率
            result_idvc = daily_ret.copy()  # 创建回归结果，结构与个股收益率一致
            result_idvc.iloc[:, :] = np.nan
            rf = pd.DataFrame(df_rev['BOND_RATE'])  # 无风险利率
            market_and_rf = market_ret.merge(rf, on=['date'])
            rf = np.array(market_and_rf.iloc[:, -1])
            x = np.array(market_and_rf.iloc[:, -2]) - rf  # 设计自变量结构
            x = sm.add_constant(x)  # 加入截距项
            for j in range(result_idvc.shape[1]):
                y = np.array(daily_ret.iloc[:, j]) - rf
                model = RollingOLS(y, x, window=120, min_nobs=60).fit()
                result_idvc.iloc[:, j] = np.sqrt(
                model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
            self.df_char_dict[stock_code]['idvc'] = result_idvc

    def calculate_financial_metrics(self):
        """为每只股票计算财务指标"""
        for stock_code in tqdm(self.df_rev_dict.keys(), desc="Calculating Financial Metrics"):
            df_rev = self.df_rev_dict[stock_code]
            rna = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPPROFIT'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV']
            self.df_char_dict[stock_code]['rna'] = rna.reindex(self.df_char_dict[stock_code].index, method='ffill')

            tot_rev = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].rolling(4).sum()[3::4]
            asset_y = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'].reindex(tot_rev.index)
            ato = tot_rev / asset_y.shift(1)
            self.df_char_dict[stock_code]['ato'] = ato.reindex(self.df_char_dict[stock_code].index, method='ffill')

            tot_opp = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPPROFIT'].rolling(4).sum()[3::4]
            tot_netp = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'].rolling(4).sum()[3::4]
            tbi = tot_opp / tot_netp
            self.df_char_dict[stock_code]['tbi'] = tbi.reindex(self.df_char_dict[stock_code].index, method='ffill')

            last_yearday = df_rev.groupby('year')['date'].idxmax()
            last_equity = df_rev.loc[last_yearday]['TOT_EQUITY']
            last_mv = df_rev.loc[last_yearday]['MKT_CAP_ARD']
            bl = last_equity / last_mv
            self.df_char_dict[stock_code]['bl'] = bl.reindex(self.df_char_dict[stock_code].index, method='ffill')
            self.df_char_dict[stock_code]['bl'] = self.df_char_dict[stock_code]['bl'].shift(1)

            last_day = df_rev.groupby('year_month')['date'].idxmax()
            dm = df_rev.loc[last_day]['TOT_LIAB'] / df_rev.loc[last_day]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['dm'] = dm

            am = df_rev.loc[last_day]['TOT_ASSETS'] / df_rev.loc[last_day]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['am'] = am

            tot_netp = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'].rolling(4).sum()[
                       3::4].reindex(self.df_char_dict[stock_code].index, method='ffill')
            ep = tot_netp / df_rev.loc[last_day]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['ep'] = ep

            ag = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'][3::4].pct_change()
            self.df_char_dict[stock_code]['ag'] = ag.reindex(self.df_char_dict[stock_code].index, method='ffill')

            tot_cfo = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_CASH_FLOWS_OPER_ACT'].rolling(4).sum()[
                      3::4]
            tot_cfo.index = pd.to_datetime(tot_cfo.index)
            ocfp = tot_cfo / df_rev.loc[tot_cfo.index]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['ocfp'] = ocfp.reindex(self.df_char_dict[stock_code].index, method='ffill')

            pm = df_rev.loc[last_day]['QFA_OPPROFIT'] / df_rev.loc[last_day]['QFA_TOT_OPER_REV']
            self.df_char_dict[stock_code]['pm'] = pm.reindex(self.df_char_dict[stock_code].index, method='ffill')

            sg = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].rolling(4).sum()[3::4].pct_change()
            self.df_char_dict[stock_code]['sg'] = sg.reindex(self.df_char_dict[stock_code].index, method='ffill')

            sgq = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].shift(4)
            self.df_char_dict[stock_code]['sgq'] = sgq.reindex(self.df_char_dict[stock_code].index, method='ffill')

            tes = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_TAX'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_TAX'].shift(4)
            self.df_char_dict[stock_code]['tes'] = tes.reindex(self.df_char_dict[stock_code].index, method='ffill')

            rs_4 = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_TOT_OPER_REV'] - \
                   df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_TOT_OPER_REV'].shift(4)
            rs_7 = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_TOT_OPER_REV'].rolling(7).std()
            rs = rs_4 / rs_7
            self.df_char_dict[stock_code]['rs'] = rs.reindex(self.df_char_dict[stock_code].index, method='ffill')

            sue_4 = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'] - \
                    df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'].shift(4)
            sue_7 = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'].rolling(7).std()
            sue = sue_4 / sue_7
            self.df_char_dict[stock_code]['sue'] = sue.reindex(self.df_char_dict[stock_code].index, method='ffill')

            ala = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_CUR_ASSETS'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'].shift(4)
            self.df_char_dict[stock_code]['ala'] = ala.reindex(self.df_char_dict[stock_code].index, method='ffill')

            liq = df_rev.loc[last_day]['MONETARY_CAP'] + 0.715 * df_rev.loc[last_day]['ACCT_RCV'] + \
                  0.547 * df_rev.loc[last_day]['INVENTORIES'] + 0.535 * df_rev.loc[last_day]['FIX_ASSETS']
            tan = liq / df_rev.loc[last_day]['TOT_ASSETS']
            self.df_char_dict[stock_code]['tan'] = tan

            dsi = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].pct_change() - \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['INVENTORIES'].pct_change()
            self.df_char_dict[stock_code]['dsi'] = dsi.reindex(self.df_char_dict[stock_code].index, method='ffill')

            dgs = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_GROSSMARGIN'].pct_change() - \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].pct_change()
            self.df_char_dict[stock_code]['dgs'] = dgs.reindex(self.df_char_dict[stock_code].index, method='ffill')

            sga = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_SELLING_DIST_EXP'] + \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_GERL_ADMIN_EXP']
            dss = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_OPER_REV'].pct_change() - sga.pct_change()
            self.df_char_dict[stock_code]['dss'] = dss.reindex(self.df_char_dict[stock_code].index, method='ffill')

            rds = df_rev.loc[last_day]['QFA_RD_EXP'] / df_rev.loc[last_day]['QFA_OPER_REV']
            self.df_char_dict[stock_code]['rds'] = rds

            rdm = df_rev.loc[last_day]['QFA_RD_EXP'] / df_rev.loc[last_day]['MKT_CAP_ARD']
            self.df_char_dict[stock_code]['rdm'] = rdm

            ch_eq = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_EQUITY'][::4].shift(1) - \
                    df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_EQUITY'][::4].shift(2)
            dbe = ch_eq / df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'][::4].shift(2)
            self.df_char_dict[stock_code]['dbe'] = dbe.reindex(self.df_char_dict[stock_code].index, method='ffill')

            self.df_char_dict[stock_code]['noa'] = df_rev.loc[last_day]['NETWORKINGCAPITAL']

            dnoa = (df_rev[(df_rev['STM_ISSUINGDATE'] == 'revised')][::4]['NETWORKINGCAPITAL'].shift(1) - \
                    df_rev[(df_rev['STM_ISSUINGDATE'] == 'revised')][::4]['NETWORKINGCAPITAL'].shift(2)) / \
                   df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'][::4].shift(2)
            self.df_char_dict[stock_code]['dnoa'] = dnoa.reindex(self.df_char_dict[stock_code].index, method='ffill')

            roe = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_EQUITY'].shift(1)
            roe.index = pd.to_datetime(roe.index)
            self.df_char_dict[stock_code]['roe'] = roe.reindex(self.df_char_dict[stock_code].index, method='ffill')

            droe = roe - roe.shift(4)
            self.df_char_dict[stock_code]['droe'] = droe.reindex(self.df_char_dict[stock_code].index, method='ffill')

            roa = df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['QFA_NET_PROFIT_IS'] / \
                  df_rev[df_rev['STM_ISSUINGDATE'] == 'revised']['TOT_ASSETS'].shift(1)
            roa.index = pd.to_datetime(roa.index)
            self.df_char_dict[stock_code]['roa'] = roa.reindex(self.df_char_dict[stock_code].index, method='ffill')

            droa = roa - roa.shift(4)
            self.df_char_dict[stock_code]['droa'] = droa.reindex(df_char.index, method='ffill')

    def run_analysis(self):
        """运行完整分析流程"""
        self.fetch_data()
        self.preprocess_data()
        self.calculate_characteristics()
        self.calculate_financial_metrics()
        return self.df_char_dict

    def save_to_csv(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建文件夹: {self.output_dir}")

        for idx, (stock_code, df_char) in enumerate(self.df_char_dict.items(), 1):
            clean_stock_code = stock_code.replace(".", "_")
            filename = os.path.join(self.output_dir, f"char_{clean_stock_code}.csv")
            # 添加 asset 列
            df_char['asset'] = stock_code
            df_char.to_csv(filename)
            print(f"数据已保存至 {filename}")
        print(f"共保存 {len(self.df_char_dict)} 只股票的数据到 {self.output_dir}")


# 示例用法
if __name__ == "__main__":
    w.start()
    sse50_stocks = w.wset("sectorconstituent", "date=2025-03-10;windcode=000016.SH").Data[1]

    # 初始化，指定输出文件夹为"stock_data"
    analyzer = StockAnalyzer(stock_codes=sse50_stocks, start_date="2010-01-01", end_date="2025-03-10",
                             output_dir="C:/jiaolong/stocks_data")

    try:
        df_char_dict = analyzer.run_analysis()
        analyzer.save_to_csv()
        for stock_code, df_char in df_char_dict.items():
            print(f"{stock_code} 前5行数据：\n{df_char.head()}")
    except SystemExit:
        print("程序已因错误终止，部分数据已保存")
    except Exception as e:
        print(f"发生未预期错误: {e}")
        analyzer.save_to_csv()