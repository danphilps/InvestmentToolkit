# package for working with tabular data
import pandas as pd
import numpy as np

# package for navigating the operating system
import os
import multiprocessing

# Progress bar
from tqdm.notebook import tqdm

# Pretty dataframe printing for this notebook
from IPython.display import display

# Suppress warnings for demonstration purposes...
import warnings

#charts
import matplotlib.ticker as mtick
import seaborn as sns
import matplotlib.pyplot as plt

# Type checking
import numbers

#Stats and math
import math
from sklearn.metrics import r2_score
from scipy.stats import f

from numpy import random
from scipy import stats

from sklearn.metrics import r2_score
from scipy.stats import f
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
import statsmodels
from statsmodels.regression.linear_model import OLS

# Functions for EDA
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import shapiro

# SAI...
from investsai.sai import SAI

#Other...
import itertools



class SimulationUtils():
    import matplotlib
    import matplotlib.pyplot as plt
    import math

    from scipy import stats

    # get the first period trades are present in
    def start_period_trades(df_trades: pd.DataFrame) -> int:

      #Step through time... earliest to latest-forecast_ahead.
      t_first_trade = -1
      for t in range(df_trades.shape[0]-1, 0, -1):
        if round(df_trades.iloc[t,:].sum(), 1) == 1:
          t_first_trade = t
          break

      return t_first_trade

    # Simulation routine
    @staticmethod
    def run_sim(df_trades: pd.DataFrame,
                df_sec_rets: pd.DataFrame,
                rebalance_freq: int = 6,
                transaction_costs: float = 0.001,
                print_chart: bool = True,
                date_start_of_sim: int = -1,
                warnings_off: bool = False) -> (pd.DataFrame, object):

        '''
        Run a historic simulation of full rebalance portfolios
        Args:
          df_trades: DataFrame with trades to incept at the frequency rebalance_freq
          df_sec_rets: stock level returns monthly, our y variable
          rebalance_freq: how often to calculate quantile trades?
          transaction_costs: costs to apply to changes in position
          print_chart: show analytic diagnostic information
          date_start_of_sim: start date from which to start. Continue to the end of the data.
        Returns:
            df_sec_cagr: expected return forecasts
            plt: return the plot object so we can add other series to it - only works if we are printing a chart
          '''

        # Sanity
        # Period1 must have a 100% allocation...
        if ((df_trades.sum(axis=1).max() > 1.001 == True) | (df_trades.sum(axis=1).max() < 99.999 == True)):
            raise TypeError("Trades pass in do not all sum to 100%")
        if df_trades.index[0] < df_trades.index[-1]:
            raise TypeError("df_trades needs to have the latest date at the top")
        if date_start_of_sim > df_trades.shape[0]:
            raise TypeError("date_start_of_sim > df_trades.shape[0]-2")

        if date_start_of_sim == -1:
            date_start_of_sim = df_trades.shape[0] - 2

        # Ini CAGR of each security as the initial trade
        df_sec_cagr = pd.DataFrame(np.zeros((date_start_of_sim + 2, df_trades.shape[1])), columns=df_trades.columns,
                                   index=df_trades.index[0:date_start_of_sim + 2])
        df_sec_cagr.astype(float)
        df_sec_cagr.columns = df_trades.columns
        df_sec_cagr = df_sec_cagr.astype(float)
        df_trades = df_trades.astype(float)
        cagr_total = -1
        period_no = -1

        # Progress
        if warnings_off == False:
            pbar = tqdm()
            pbar.reset(total=date_start_of_sim)  # initialise with new `tota

        # Step through time: rebalance_freq
        for t in range(date_start_of_sim, -1, -1):
            if warnings_off == False:
                # Progress
                pbar.update()

            # Trades in period?
            if (df_trades.iloc[t, :].sum() != 0):
                # Start positions
                # ===============================================
                if cagr_total == -1:
                    df_sec_cagr.iloc[t, :] = df_trades.iloc[t, :]
                    cagr_total = df_sec_cagr.iloc[t, :].sum(skipna=True)
                    period_no = 0

                    # Sanity
                    if (round(cagr_total, 2) != 1):
                        raise TypeError(
                            "Initial trades do not equal 100% Row: " + str(i) + ', total:' + str(round(cagr_total, 2)))

                else:
                    # Establish any buy trades in the period...
                    # ================================================
                    trade_buy_flag = ((df_trades.iloc[t, :] != np.nan) & (df_trades.iloc[t, :] > 0)).to_list()
                    trade_buy_cols = df_trades[df_trades.columns[trade_buy_flag]]
                    tot_buy_trades = df_trades[df_trades.columns[trade_buy_flag]].iloc[t, :].sum(skipna=True)

                    # Sanity: All trades are assumed to be rebalances of 100% of the portfolio
                    if round(tot_buy_trades, 2) != 1:
                        raise TypeError('Trades must equal 100% not:')
                    if df_trades.columns[trade_buy_flag].__len__() == 0:
                        raise TypeError('No valid trades')

                        # Positions roll forwards...
                    # ================================================
                    df_sec_cagr.iloc[t, :] = df_sec_cagr.iloc[t + 1, :] * (1 + df_sec_rets.iloc[t, :])
                    cagr_total = df_sec_cagr.iloc[t, :].sum()

                    # Execute Sell/Buys instantaneously at the close.
                    # (We overwrite the CAGR of the buy trades with the new trade* cagr_total,
                    # and we zero out the sell trades, having already added their contribution to the cagr_total)
                    # ================================================
                    # Remove all positions
                    df_sec_cagr.iloc[t, :] = 0
                    # df_sec_cagr[df_sec_cagr.columns[trade_buy_flag]].iloc[t,:] = df_trades[df_trades.columns[trade_buy_flag]].iloc[t,:].copy() * cagr_total
                    #for j in range(df_sec_cagr.shape[1]):
                    #    if trade_buy_flag[j]:
                    #        df_sec_cagr.iloc[t, j] = df_trades.iloc[t, j] * cagr_total
                    df_sec_cagr.iloc[t, trade_buy_flag] = df_trades.iloc[t, trade_buy_flag] * cagr_total

                    # Sanity
                    if len(df_trades.columns[trade_buy_flag]) == 0:
                        raise TypeError('df_trades.columns[trade_buy_flag].__len__() == 0')
                    if len(df_sec_cagr.columns[trade_buy_flag]) == 0:
                        raise TypeError('f_sec_cagr.columns[trade_buy_flag].__len__() == 0')

                        # Transaction costs
                    # trans_costs_current = abs(df_trades[df_trades.columns[trade_buy_flag]].iloc[t,:] - df_sec_cagr[df_sec_cagr.columns[trade_buy_flag]].iloc[t,:]/cagr_total)
                    # trans_costs_current_applied = df_trades[df_trades.columns[trade_buy_flag]].iloc[t,:] * -abs(trans_costs_current)
                    # trans_costs_current_applied = trans_costs_current_applied.copy(deep=True)
                    # df_sec_cagr[df_sec_cagr.columns[trade_buy_flag]].iloc[t,:] += trans_costs_current_applied
                    trans_costs_current = abs(df_trades.iloc[t, :] - df_sec_cagr.iloc[t, :] / cagr_total)
                    for j in range(df_sec_cagr.shape[1]):
                        if trade_buy_flag[j]:
                            df_sec_cagr.iloc[t, j] += df_trades.iloc[t, j] * -abs(trans_costs_current.iloc[j])
                    #df_sec_cagr.iloc[t, trade_buy_flag] += df_trades.iloc[t, trade_buy_flag] * -abs(trans_costs_current.iloc[trade_buy_flag])

                    period_no += 1
            else:
                # No trades... Positions roll forwards...
                # ================================================
                # Sanity
                if df_sec_rets.iloc[t, :].abs().sum(skipna=True) == 0:
                    raise TypeError("Returns are zero in period: " + str(i))
                    # df_sec_cagr.iloc[t,:] = df_sec_cagr.iloc[t+1,:] * (1+df_sec_rets.iloc[t,:])
                for j in range(df_sec_cagr.shape[1]):
                    df_sec_cagr.iloc[t, j] = df_sec_cagr.iloc[t + 1, j] * (1 + df_sec_rets.iloc[t, j])
                #df_sec_cagr.iloc[t, :] = df_sec_cagr.iloc[t + 1, :] * (1 + df_sec_rets.iloc[t, :])

                period_no += 1

        # Sort into date order
        df_sec_cagr = df_sec_cagr.sort_index(ascending=False)

        # Analytics:
        if print_chart == True:
            p = SimulationUtils.sim_chart(df_sec_cagr=df_sec_cagr)
            p.show()
        else:
            p = None

        return (df_sec_cagr, p)

    @staticmethod
    def sim_chart(df_sec_cagr: pd.DataFrame):
        '''
        Print return based analytics and chart CAGR
        Args:
          df_sec_cagr: DataFrame continaing compound annual growth series of the strategy to be charted
          period_no: Number of periods in the above series to chart/analyze.
        Returns:
            None
          '''

        # ini
        df_final = df_sec_cagr.sum(axis=1, skipna=True)
        period_count = [x for x in range(df_final.shape[0] - 1, 0, -1) if df_final.iloc[x] != 0].__len__() + 1
        df_final = df_final.iloc[0:period_count].astype(float)
        df_final_mom_tr = pd.DataFrame(df_final.iloc[:-1].values / df_final.iloc[1:].values - 1, index=None)

        # Analytics
        sim_ana = pd.DataFrame(np.zeros((4, 1)), index=None)
        idx = ['Total Return (TR)(annualized)']
        sim_ana.iloc[0] = df_final.iloc[0] ** (12 / period_count) - 1
        idx.append('Standard deviation (SD) (annualized)')
        sim_ana.iloc[1] = df_final_mom_tr.std() * math.sqrt(12)
        idx.append('Sharpe Ratio (TR/SD)')
        sim_ana.iloc[2] = float(sim_ana.iloc[0, 0]) / float(sim_ana.iloc[1, 0])
        idx.append('Hit rate (monthly)')
        sim_ana.iloc[3] = df_final_mom_tr[df_final_mom_tr >= 0].iloc[:, 0].count() / df_final_mom_tr[
                                                                                         df_final_mom_tr != 0].iloc[:,
                                                                                     0].count()
        sim_ana.columns = ['Analytics']
        sim_ana.index = idx
        display(sim_ana)

        # Descriptive stats
        sim_ana2 = pd.DataFrame(np.zeros((5, 1)), index=None)
        idx = ['Mean (TR)']
        sim_ana2.iloc[0] = df_final_mom_tr.mean()
        idx.append('Median (TR)')
        sim_ana2.iloc[1] = df_final_mom_tr.median()
        idx.append('Variance (TR)')
        sim_ana2.iloc[2] = df_final_mom_tr.var()
        idx.append('Skewness (TR)')
        sim_ana2.iloc[3] = stats.skew(df_final_mom_tr)
        idx.append('Kurtosis (TR)')
        sim_ana2.iloc[4] = stats.kurtosis(df_final_mom_tr)
        sim_ana2.columns = ['Descriptive Stats']
        sim_ana2.index = idx
        display(sim_ana2)

        # CAGR chart
        plt.figure(figsize=(20, 10))
        plt.plot([t for t in range(0, period_count)], df_final.iloc[0:period_count])
        plt.xticks([t for t in range(0, period_count)], df_final.index[0:period_count])
        ax = plt.gca()
        ax.invert_xaxis()
        plt.xticks(rotation=90)
        plt.title('Compound Annual Growth: Investment Strategy')
        plt.xlabel('Date')
        plt.ylabel('CAGR where start period = 1')
        plt.grid()

        return plt

    @staticmethod
    def sim_chart_add_series(p: object,
                             df_cagr: pd.DataFrame,
                             series_name: str,
                             emphasize: bool = False):
        '''
        Add series to existing sim_chart(...,print_chart=True) CAGR chart
        Args:
          p: plt object created from sim_chart
          df_cagr: DataFrame continaing compound annual growth series of the strategy to be charted
          title: ... name of this return series
        Returns:
            None
        '''

        # Sanity
        if p is None:
            raise TypeError(
                'p object is none. Need to pass the 2nd return object from run_sim to this function. Remember to set print_chart=True in the call to run_sim')

        # Get the dimensions of the x-axis of p
        ax = p.gca()
        line = ax.lines[0]
        first_value = [x for x in range(line.get_ydata().__len__() - 1, 0, -1) if
                       line.get_ydata(0)[x] != 0].__len__() + 1

        # Add series
        if emphasize == True:
            p.plot([t for t in range(0, first_value)], df_cagr.iloc[0:first_value], label=series_name, linewidth=3)
        else:
            p.plot([t for t in range(0, first_value)], df_cagr.iloc[0:first_value], label=series_name, linewidth=1)

        return

    # get the first period trades are present in
    @staticmethod
    def start_period_trades(df_trades: pd.DataFrame) -> int:

        # Step through time... earliest to latest-forecast_ahead.
        t_first_trade = -1
        for t in range(df_trades.shape[0] - 1, 0, -1):
            if round(df_trades.iloc[t, :].sum(), 1) == 1:
                t_first_trade = t
                break

        return t_first_trade

    from tqdm.notebook import tqdm
    # Establish trades for 6monthly rebalances: Return a full DF of trade weights (per column/security)
    # For every time slice, set the highest quantile as equal weighted trades
    # Quantiles have nan entries and zero entries removed
    @staticmethod
    def trades_topquantile_generate(df_all_er: pd.DataFrame,
                                    rebalance_freq: int,
                                    min_quantile_to_buy: float = 0) -> pd.DataFrame:

        '''
        Generate simple trades, >=min_quantile_to_buy of df_all_er. Equal weight the stocks selected
        Args:
          df_all_er: ALl stock and timeperiod expected returns
          rebalance_freq: how often to calculate quantile trades?
          min_quantile_to_buy: >=quantile to model equal weighted trades
        Returns:
            df_stock_er: expected return forecasts
        '''

        # Sanity
        if df_all_er.index[1] < df_all_er.index[2]:
            raise TypeError("Sort order of dates is wrong. Latest date should be at the top.")
        if (rebalance_freq < 0 == True) | (rebalance_freq > 60 == True):
            raise TypeError("rebalance_freq < 0 | rebalance_freq > 60.")
        if (min_quantile_to_buy < 0 == True) | (min_quantile_to_buy > 1 == True):
            raise TypeError("min_quantile_to_buy < 0 | min_quantile_to_buy > 1")

        # Initialise df_trades
        df_trades = pd.DataFrame(np.zeros((df_all_er.shape[0], df_all_er.shape[1])), columns=df_all_er.columns,
                                 index=df_all_er.index)
        insert_zero_row = False  # Error? or zero values? We insert a zero row...
        date_start_non_zero = -1  # What is the first non-zero row

        # Progress
        pbar = tqdm()
        pbar.reset(total=int((df_all_er.shape[0] - 1) / rebalance_freq))  # initialise with new `total`

        # Loop through time... start to end.
        # Progress bar
        for t in range(df_all_er.shape[0] - 1, 0, -rebalance_freq):
            # Progress
            pbar.update()

            # Get this period's stocks (cols), get valid stocks only
            df_opset_curr = pd.DataFrame(df_all_er.iloc[t, :], index=None).T
            valid_secs = (df_opset_curr.iloc[0, :].isna() | df_opset_curr.iloc[0, :] != 0).to_numpy()

            # Get our securities above the min_quantile_to_buy
            qu = df_opset_curr[df_opset_curr.columns[valid_secs]].quantile(q=min_quantile_to_buy, axis=1).iloc[0]
            # Capture all valid securities if min_quantile_to_buy==0 (ie universe securities)
            if min_quantile_to_buy == 0:
                qu = qu - 1
            # How many securities is that?
            no = df_opset_curr[df_opset_curr[df_opset_curr.columns[valid_secs]] > qu].count(axis=1).iloc[0]

            # Create trades row
            df_trades_curr = pd.DataFrame(df_all_er.iloc[t, :], index=None).T
            df_trades_curr.astype(float)
            df_trades_curr.iloc[:, :] = float(0)
            if no != 0:
                # Add equal weighted trades...
                trade_secs = ((df_opset_curr.iloc[0, :] > qu) & (df_opset_curr.iloc[0, :] != 0))
                df_trades_curr[df_trades_curr.columns[trade_secs]] = 1 / no

                # Record the first non-zero row
                if date_start_non_zero == -1:
                    date_start_non_zero = t

                insert_zero_row = False
            else:
                insert_zero_row = True

            # Insert zero row...
            if insert_zero_row == True:
                df_trades_curr = pd.DataFrame(df_all_er.iloc[t, :], index=None).T
                df_trades_curr.iloc[:, :] = float(0.00000)

            # Overwrite date/time row in df_trades
            df_trades = df_trades.drop(axis=0, index=df_trades_curr.index)
            df_trades = pd.concat((df_trades, df_trades_curr), axis=0)

            # Correct the sort order...
        df_trades = df_trades.sort_index(ascending=False)

        # Sanity... Trades in each period totals a max of 100%? if non-zero.
        if ((df_trades.sum(axis=1).max() > 1.001 == True) | (df_trades.sum(axis=1).max() < 99.999 == True)):
            raise TypeError("Sanity checks on returns failed")

        # Progress
        print('Top quantile generated')

        return df_trades

    @staticmethod
    def stock_rets_get(df_tb3ms: pd.DataFrame,
                       df_sec_rets: pd.DataFrame,
                       date_start: int,
                       date_end: int,
                       window_size: int = 36) -> pd.DataFrame:
        '''
        Calculate security returns on the same term as the factor loadings.
        This will be the input for our ML model.
        for securities with zero values in their history, remove them
        Args:
            df_tb3ms: DataFrame containing time series of 3m risk free rate
            ols_betas_final: DataFrame containing factor loadings for all securities
            date_start: training time window start period
            date_end: training time window end period
            window_size: Factor lodings calculated over what period?
        Returns:
            df_all_stock_returns: stock returns, cols are stocks, rows are time points between date_start and date_end; returns annualized over window_size
        '''

        # sanity
        if df_sec_rets.shape[0] < date_start + window_size:
            raise TypeError("df_sec_rets.shape[0] < date_start + window_size")
        if df_sec_rets.shape[0] < date_start:
            raise TypeError("df_sec_rets.shape[0] < date_start")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")

            # NB: elements of the dict = time periods. The element numbers corrspond
        # with row numbers in all other DatFrames in this note book (eg t=1, it t=1 in all other structures)
        df_all_stock_returns = pd.DataFrame(np.zeros((date_start - date_end, df_sec_rets.shape[1])), index=None)
        df_all_stock_returns.index = df_sec_rets.index[date_end:date_start]
        df_all_stock_returns.columns = df_sec_rets.columns

        z = 0
        for i in range(date_start, date_end, -1):
            # Calc stock returns, annualized
            sec_returns = df_sec_rets.iloc[i:i + window_size, :].sort_index(ascending=True)

            # Ignore any entries with zeros
            sec_returns = sec_returns.replace(to_replace=0, value=np.nan)
            ignore_cols = ~df_sec_rets.isna().max(axis=0)

            sec_returns = 1 + sec_returns.loc[:, ignore_cols]
            sec_returns = sec_returns.loc[:, ignore_cols].prod(axis=0)

            # Assume monthly data, annualize
            if (window_size > 12):
                # Supress warnings here...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sec_returns.loc[ignore_cols] = sec_returns.loc[ignore_cols] ** (12 / (window_size)) - 1
            else:
                sec_returns.loc[ignore_cols] = sec_returns.loc[ignore_cols] - 1

            # All stock returns...
            df_all_stock_returns.loc[df_all_stock_returns.index[z], ignore_cols] = sec_returns.T.values
            z += 1

        return df_all_stock_returns

class RobustInvestmentUtils():

    # Function we will call to add R2 and p-val to the off-diagonal cells of the pair plot
    def R2func(x, y, hue=None, ax=None, **kws):
        """Plot the correlation coefficient in the top left hand corner of a plot."""
        _, _, r, p, _ = stats.linregress(x, y)
        ax = ax or plt.gca()
        ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
        ax.annotate(f'p-val = {p:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

    # Function we will call to add normality test stat and p-val to diagnonal cells of pair plot
    # Note that inputs to linear regression are not required to be normally distributed.
    def normalityfunc(x, hue=None, ax=None, **kws):
        """Plot the Shapiro Wilk p-value in the top left hand corner of diagonal cells."""
        stat, p = shapiro(x)
        ax = ax or plt.gca()
        ax.annotate(f'Shapiro-Wilk stat = {stat:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
        ax.annotate(f'p-val = {p:.2f}', xy=(.1, .8), xycoords=ax.transAxes)

    # Target shuffling "lite": get empirical distribution of opportunity-set
    @staticmethod
    def _target_shuffling_lite_get_dist(df_opportunity_set_trades: pd.DataFrame,
                                        min_quantile_to_buy: float,
                                        df_sec_rets: pd.DataFrame,
                                        rebalance_freq: int = 6,
                                        iterations: int = 100,
                                        show_progress: bool = True,
                                        q: multiprocessing.Queue = None) -> pd.DataFrame:

        '''
        Return the empirical distribution of possible return outcomes for the opportunity-set selected
        This can be extended to return outcomes for the learner selected
        Randomly shuffle the "buy" names, same opportunity set as the model used
        which removes securities excluded fom the models owing to NaNs in training windows etc etc
        Args:
          df_opportunity_set_trades: DataFrame with the full opportunityset of trades from the model being tested (ie min_quantile_to_buy set to 0)
          min_quantile_to_buy: what is your model selecting? to 20th percentile? Thats what we will simulate on here.
          df_sec_rets: stock level returns monthly, our y variable
          rebalance_freq: how often to calculate quantile trades?
          iterations: number of times we create and sim random portfolios.
          q: for multiprocessing of this function. Leave as None to run normally.
        Returns:
            df_sec_cagr: end cagr for each of the (iterations) sims
        '''

        # Sanity
        if min_quantile_to_buy == 0:
            raise TypeError('min_quantile_to_buy == 0')

        # Find the first trades
        for t in range(df_opportunity_set_trades.shape[0] - 1, -1, -1):
            if df_opportunity_set_trades.iloc[t, :].sum() > 0:
                t_start = t
                break

        # Ini:
        dt_sim_tr = pd.DataFrame(np.zeros((iterations, 4)), index=None)
        dt_sim_tr = dt_sim_tr.astype(float)
        # Record outcome
        dt_sim_tr.columns = ['cagr', 'tr_ann', 'sd_ann', 'sharpe']

        dt_rnd_trades = pd.DataFrame(np.zeros((df_opportunity_set_trades.shape[0], df_opportunity_set_trades.shape[1])),
                                     index=None)
        dt_rnd_trades.columns = df_opportunity_set_trades.columns
        dt_rnd_trades.index = df_opportunity_set_trades.index

        # Progress
        if show_progress == True:
            pbar = tqdm()
            pbar.reset(total=iterations)  # initialise with new `tota

        for i in range(iterations):
            # Progress
            if show_progress == True:
                pbar.update()

            # randomly allocate trades
            for t in range(t_start, -1, -rebalance_freq):
                curr_date = df_opportunity_set_trades.index[t]
                # opportunity set secs
                op_set_cols_mask = df_opportunity_set_trades.iloc[t, :] > 0
                rnd_buys = random.rand(op_set_cols_mask.sum())
                # Set random trades
                rnd_buys[rnd_buys < np.quantile(rnd_buys, min_quantile_to_buy)] = 0
                rnd_buys[rnd_buys != 0] = 1 / np.count_nonzero(rnd_buys)  # Equal weight stocks...

                # Set random trades
                dt_rnd_trades.loc[curr_date, op_set_cols_mask] = rnd_buys

                # Sanity
                if round(dt_rnd_trades.loc[curr_date, :].sum(), 1) != 1:
                    raise TypeError('dt_rnd_trades.loc[curr_date, :] != 1')

            # Run the simulation function
            df_sec_cagr, p = SimulationUtils.run_sim(df_trades=dt_rnd_trades,
                                     rebalance_freq=6,
                                     df_sec_rets=df_sec_rets,
                                     print_chart=False,
                                     warnings_off=True)

            # Calc TR, sd, Sharpe
            df_cagr = df_sec_cagr.sum(axis=1)

            period_count = [x for x in range(df_cagr.shape[0] - 1, 0, -1) if df_cagr.iloc[x] != 0].__len__()
            df_final_mom_tr = pd.DataFrame(
                df_cagr.iloc[:period_count - 1].values / df_cagr.iloc[1:period_count].values - 1, index=None)

            ret = []
            ret.append(df_cagr.iloc[0].sum())
            ret.append(df_cagr.iloc[0].sum() ** (12 / period_count) - 1)
            ret.append(df_final_mom_tr.std() * math.sqrt(12))
            ret.append(ret[1] / ret[2])

            # Buld the distribution
            dt_sim_tr.loc[i, :] = pd.Series(ret).astype(float).values

        # multiprocessing
        if q is not None:
            q.put(dt_sim_tr)

        return dt_sim_tr

    # Target shuffling "lite": get empirical distribution
    @staticmethod
    def target_shuffling_lite_get_dist(df_opportunity_set_trades: pd.DataFrame,
                                       min_quantile_to_buy: float,
                                       df_sec_rets: pd.DataFrame,
                                       rebalance_freq: int = 6,
                                       iterations: int = 10) -> pd.DataFrame:
        '''
        Target shuffling lite, is a "permutation testing" method, generating the empirical distribution
        of return outcomes available from  the opportunity-set of stocks available
        Multipriocessed version..
        Return the empirical distribution of possible return outcomes for the opportunity-set of stocks selected
        Randomly shuffle the "buy" names, same opportunity set as the model used
        which removes securities excluded fom the models owing to NaNs in training windows etc etc
        Args:
          df_opportunity_set_trades: DataFrame with the full opportunityset of trades from the model being tested (ie min_quantile_to_buy set to 0)
          df_sec_rets: stock level returns monthly, our y variable
          rebalance_freq: how often to calculate quantile trades?
          iterations: number of times we create and sim random portfolios.
        Returns:
            df_sec_cagr: end cagr for each of the (iterations) sims
        '''

        # Multi process...
        jobs = []
        num_cores = multiprocessing.cpu_count()
        iterations_sub = int(iterations / num_cores)

        # Progress
        pbar = tqdm()
        pbar.reset(total=num_cores)  # initialise with new `tota

        # create a queue
        q = multiprocessing.Queue()

        # Kick off processes
        for i in range(num_cores):
            proc = multiprocessing.Process(target=_target_shuffling_lite_get_dist, args=(df_benchmark_trades,
                                                                                         min_quantile_to_buy,
                                                                                         df_sec_rets,
                                                                                         rebalance_freq,
                                                                                         iterations_sub,
                                                                                         False,
                                                                                         q))
            jobs.append(proc)
            proc.start()

        # Wait for completion
        for p in jobs:
            # Progress
            pbar.update()

            p.join()

        # Get the results
        ret = []
        i = 0
        for p in jobs:
            ret.append(q.get())

        # Combine..
        final_ret = ret[0]
        for i in range(1, len(ret)):
            final_ret = pd.concat((final_ret, ret[i]), axis=0)

        return final_ret

    # Test the strategy again the target shuffling distribution
    @staticmethod
    def target_shuffling_chart(dt_target_shuffling_dist: pd.DataFrame,
                               df_sec_cagr: pd.DataFrame,
                               metric_to_test: str = 'tr_ann'):

        '''
        Test the strategy results against a "target shuffling" distribution to determine empirical
        p-value of the returns.
        generates charts and stats to determine the significance of simulation results
        Args:
          dt_target_shuffling_dist: df produced by the function target_shuffling_get_dist (columns... ['cagr', 'tr_ann', 'sd_ann', 'sharpe'])
          dt_sim_tr: CAGR series of the strategy to test from the function run_sim
          metric_to_test: which metric to test? ['cagr', 'tr_ann', 'sd_ann', 'sharpe']
        Returns:
            None
        '''

        # sanity:
        if df_sec_cagr.shape[1] == 0:
            raise TypeError('Expecting df_sec_cagr that contains columns for multiple stocks')

        # Ini
        dt_sim_tr = df_sec_cagr.sum(axis=1)

        # strategy: cagr seriesm, get analytrics
        period_count = [x for x in range(dt_sim_tr.shape[0] - 1, 0, -1) if dt_sim_tr.iloc[x] != 0].__len__()
        df_final_mom_tr = pd.DataFrame(
            dt_sim_tr.iloc[:period_count - 1].values / dt_sim_tr.iloc[1:period_count].values - 1, index=None)

        df_ret = pd.DataFrame(np.zeros((1, 4)), index=None)
        df_ret.iloc[0, 0] = dt_sim_tr.iloc[0]
        df_ret.iloc[0, 1] = dt_sim_tr.iloc[0] ** (12 / period_count) - 1
        df_ret.iloc[0, 2] = df_final_mom_tr.values.std() * math.sqrt(12)
        df_ret.iloc[0, 3] = df_ret.iloc[0, 1] / df_ret.iloc[0, 2]
        df_ret.columns = ['cagr', 'tr_ann', 'sd_ann', 'sharpe']
        df_ret.astype(float)

        # Chart
        bins = max(min(50, int(dt_target_shuffling_dist.shape[0] / 10)), 10)

        # upper and lower CIs
        upper_ci = dt_target_shuffling_dist[metric_to_test].quantile(q=0.90)
        lower_ci = dt_target_shuffling_dist[metric_to_test].quantile(q=0.10)
        strategy_score = float(df_ret[metric_to_test])

        plt.clf()
        # plt.figure(figsize=(10,10))
        plt.hist(dt_target_shuffling_dist[metric_to_test], bins=bins)
        plt.gca().set_title(
            'Empirical Distribution Frequency Bar Chart (Target Shuffling): ' + metric_to_test + '(iterations = ' + str(
                dt_target_shuffling_dist.shape[0]) + ')', wrap=True)
        plt.axvline(x=upper_ci, color='red', label='Upper CI (90%)')
        plt.axvline(x=lower_ci, color='red', label='Lower CI (10%)')
        plt.axvline(x=strategy_score, color='black', label='Strategy Value', linewidth=10)
        plt.legend()
        plt.xlabel(metric_to_test)
        plt.ylabel('frequency')
        plt.show()

        # Stats
        p_val = stats.percentileofscore(dt_target_shuffling_dist[metric_to_test].T.to_numpy(), strategy_score) / 100
        print('Empirical probability value of the strategy: ' + metric_to_test + ':' + str(round((1-p_val), 2)))

        return

    # ***********************************************************************
    # *** Complexity bias Sanity Check!***
    # ***********************************************************************
    @staticmethod
    def bias_complexity_check_regression(no_of_instances: int,
                                         no_of_features: int = 0,
                                         no_of_parameters: int = 0) -> (bool, int, int):

        '''
        Check the complexity of the mode based on rules of thumb.
        Args:
          no_of_instances: Number of rows in your dataset
          no_of_features: Number of columns
          no_of_parameters: Number of weights/coefficients/parameters in your model
        Returns:
            rf: sklearn model object
        Author:
            failed: Did the complexity check fail? Too complex...
            feature_max: maximum number of features you shluld have given the problem type and instances
            param_max: maximum number of weights/coefficients/parameters in your model given the problem type and instances
        '''

        failed = False
        cb_K = no_of_features
        cb_n = no_of_instances

        # 1. Feature complexity: n ≥ 50 K
        if cb_n > 50 * cb_K:
            failed = True

        feature_max = int(round(cb_n / 50, 0))

        # 2. Parameter complexity: ¦θ¦ ≤ n/10
        #
        # The number of model parameters (ie weights) should observe the constraint
        # wrt training instances, n, features, K:
        #
        # |theta| >= n/10
        param_max = int(round(cb_n / 10, 0))

        if no_of_parameters > param_max:
            failed = True

        return (failed, feature_max, param_max)


    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

class NonLinearFactorInvesting():

    # Function that will run our OLS model to determine factor loadings, for a given security,
    # over a given period
    # Note the two optional parameter...
    #   use_robust_cm: estimate from a robust covariance matrix
    #   plot_residual_scatter: which will generate a scatter plot of our residuals (y vs y_hat)
    @staticmethod
    def factormodel_train_single_security(sec_col_no: int,
                         df_tb3ms: pd.DataFrame,
                         df_sec_rets: pd.DataFrame,
                         df_ff_factors: pd.DataFrame,
                         date_start: int,
                         date_end: int,
                         use_robust_cm: bool = False,
                         plot_residual_scatter: bool = False) -> (object, np.array, np.array):

      '''
      Calculate the factor loadings of a single security

      Args:
        sec_col_no: Security to select, row number in our dataframe
        df_tb3ms: Risk free rate timeseries
        df_sec_rets: stock level returns monthly, our y variable
        df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series, our X variables.
        date_start: training time window start period
        date_end: training time window end period
        use_robust_cm: use robust standard errors
        plot_residual_scatter: generate a graph of the residulas for the model

      Returns:
          ols_model: OLS, sklearn model object
          y: y variable used
          y_hat: in sample forecast of y variable.

      '''

      # sanity
      if date_start < date_end:
        raise TypeError("Latest date is date=0, date_start is > date_end")
      if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
        raise TypeError("Must pass factor returns as columns not rows")
      if df_ff_factors.index[0] != df_sec_rets.index[0]:
        raise TypeError("Dates misaligned")
      if df_tb3ms.index[0] != df_sec_rets.index[0]:
        raise TypeError("Dates misaligned")

      # Get X and y data...
      # NB: Security returns... deduct Rf
      y = [df_sec_rets.iloc[t, sec_col_no] - df_tb3ms.iloc[t, 0] for t in range(date_end,date_start)]
      X = df_ff_factors.iloc[date_end:date_start, :]

      # Instantiate and train OLS model
      # We will leave the inputs unaltered but if we normalized, it would result in
      # an intercept of aproximately zero, making forecasting down to the stock level betas
      X = sm.add_constant(X) #<< statsmodels requires we manually add an intercept.
      ols_model = OLS(y, X)
      ols_model = ols_model.fit()

      # Optional ... Use heteroskedasticity-autocorrelation robust covariance?
      if use_robust_cm:
          ols_model = ols_model.get_robustcov_results()

      # Predict in sample
      y_hat = ols_model.predict(X)
      resid = y-y_hat

      # Optional ...
      if plot_residual_scatter == True:

        # In sample prediction: Examine residuals for normality...
        # NB: The null hypothesis (H0) states that the variable is normally distributed,
        # and the alternative hypothesis (H1) states that the variable is NOT normally distributed.
        sw_stat, sw_p = shapiro(resid)
        # Check for normality in the residuals
        if sw_p < 0.10:
          print("Residuals appear to be non-normal!") # Make an error in live code

        # square plot
        #plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y-y_hat, y_hat)
        plt.title('Residual Plot: Shapiro-Wilk p-val: ' + str(round(sw_p, 2)))
        plt.show()

      return (ols_model, y, y_hat)

    @staticmethod
    def nonlinfactor_er_func_prep_data(df_tb3ms: pd.DataFrame,
                                       df_sec_rets: pd.DataFrame,
                                       df_ff_factors: pd.DataFrame,
                                       date_end: int,
                                       func_training_period: int = 1,
                                       forecast_ahead: int = 6,
                                       window_size: int = 36) -> (pd.DataFrame, pd.DataFrame):

        '''
        Prepare data for training, testing and predicting from the nn.
        **For training forecast_ahead > 0**
        **For prediction forecast_ahead = 0**
        NB: This look ahead from date_end! Data snooping risk!!
        Args:
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            date_end: training time window end period
            func_training_period: pass 1 for predictions, >=1 for training. How many periods to use to train the nn? func_training_period=1 will use only one cross section, t=date_end
            forecast_ahead: how many periods ahead are we predicting. Set this to 0 if we need data to predict.
            window_size: return window to use when calculating stock and factor returns.
        Returns:
            X: X data used to train/test/predict
            y: X data used to train/test/predict
        '''

        # sanity
        if func_training_period < 1:
            raise TypeError("func_training_period <1")
        if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
            raise TypeError("Must pass factor returns as columns not rows")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if df_ff_factors.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if (func_training_period < 0) | (func_training_period > df_sec_rets.shape[0]):
            raise TypeError("(func_training_period < 0) | (func_training_period > df_sec_rets.shape[0]")
        if (window_size < 0) | (window_size > df_sec_rets.shape[0]):
            raise TypeError("(window_size < 0) | (window_size > df_sec_rets.shape[0]")

            # Built training data over thei many periods: func_training_period
        # The time points to load are limited by the dtaa we have. Build X and y for window_size, func_training_period, else the longest period available...
        X = pd.DataFrame()
        y = pd.DataFrame()
        for t in range(date_end + forecast_ahead, date_end + forecast_ahead + func_training_period):

            # ================================
            # Get data components for X at time t: df_stock_factor_loadings, factor_excess_returns;
            # and y: stock_returns
            # ================================
            # X data..
            stock_factor_loadings, _, _ = LinearFactorInvesting.factormodel_train_manysecurities(df_tb3ms=df_tb3ms,
                                                                           df_sec_rets=df_sec_rets,
                                                                           df_ff_factors=df_ff_factors,
                                                                           date_start=t + window_size,
                                                                           # << Note we pass in the start date here
                                                                           date_end=t,
                                                                           test_complexity=False)  # << Note we pass in the end date here

            # X data... Factor returns to assume
            factor_excess_returns = df_ff_factors.iloc[t - forecast_ahead:t + window_size - forecast_ahead,
                                    :].sort_index()
            factor_excess_returns = np.array(1 + factor_excess_returns).prod(axis=0)
            factor_excess_returns.astype(float)
            # Annualize
            if (window_size > 12):
                # Supress warnings here...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    factor_excess_returns = factor_excess_returns ** (12 / (window_size)) - 1

                    # Deduct rf from stock level returns, as it will have been for factor returns
            rf_ret = df_tb3ms.iloc[t - forecast_ahead:t + window_size - forecast_ahead, :].sort_index()
            rf_ret.astype(float)

            # y data... Stock level returns ... forecast ahead by forecast_ahead periods
            stock_returns = df_sec_rets.iloc[t - forecast_ahead:t + window_size - forecast_ahead, :].sort_index()
            stock_returns.astype(float)

            # Adjust returns for r_fs
            for j in range(0, stock_returns.shape[1] - 1):
                stock_returns.iloc[:, j] = (1 + stock_returns.iloc[:, j].values) / (1 + rf_ret.iloc[:, 0].values) - 1  # subtract r_f from each monthly return
            #stock_returns.iloc[:, :] = (1 + stock_returns.iloc[:, :].values) / (1 + rf_ret.iloc[:, 0].values) - 1  # subtract r_f from each monthly return
            stock_returns = np.array(1 + stock_returns).prod(axis=0)

            # Annualize
            if (window_size > 12):
                # Supress warnings here...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stock_returns = stock_returns ** (12 / (window_size)) - 1

            # Prep X and y at time for training nn
            # ================================
            # X...
            # Dimension X
            X_t = pd.DataFrame(
                np.zeros((stock_factor_loadings.shape[0] + df_ff_factors.shape[1], stock_factor_loadings.shape[1])),
                index=None)
            # For training we can have 1 or more time period using "_t" suffix;
            # For predicting we must only have 1 time period and columns should NOT have a suffix
            if (func_training_period > 1):
                X_t.columns = [str(col) + '_' + str(t) for col in df_sec_rets.columns]
            else:
                X_t.columns = [col for col in df_sec_rets.columns]

            # Add stock loadings to X
            X_t.iloc[0:stock_factor_loadings.shape[0], :] = stock_factor_loadings.copy(
                deep=True)  # dict is indexed off 1, DFs and arrays off 0
            X_t.astype(float)
            X_index = stock_factor_loadings.index + '_coef'

            # Add factor returns to X (duplicate for each stock)
            for j in range(0, stock_factor_loadings.shape[1]):
                X_t.iloc[stock_factor_loadings.shape[0]:, j] = factor_excess_returns
            #X_t.iloc[stock_factor_loadings.shape[0]:, :] = factor_excess_returns
            X_index = X_index.append(df_ff_factors.columns + '_ret')
            X_t.index = X_index

            # y...
            y_t = pd.DataFrame(stock_returns.copy(), index=None).T  # dict is indexed off 1, DFs and arrays off 0
            # For training we can have 1 or more time period using "_t" suffix;
            # For predicting we must only have 1 time period and columns should NOT have a suffix
            if (func_training_period > 1):
                y_t.columns = [str(col) + '_' + str(t) for col in df_sec_rets.columns]
            else:
                y_t.columns = [col for col in df_sec_rets.columns]
            y_t.index = ['ALL']

            # Refine X and y
            X_t = X_t.replace(to_replace=0, value=np.nan)
            y_t = y_t.replace(to_replace=0, value=np.nan)

            X_invalid_cols_to_drop = X_t.columns[X_t.isna().sum() > 0]
            y_invalid_cols_to_drop = y_t.columns[y_t.isna().sum() > 0]
            invalid_cols_to_drop = X_invalid_cols_to_drop.append(y_invalid_cols_to_drop)
            invalid_cols_to_drop = invalid_cols_to_drop.unique()

            X_t = X_t.drop(columns=invalid_cols_to_drop)
            y_t = y_t.drop(columns=invalid_cols_to_drop)

            # Add X and y at time t to the master X, y
            X = pd.concat((X, X_t), axis=1)
            y = pd.concat([y, y_t], join='outer', axis=1)

        return X.T, y.T


    from sklearn.metrics import r2_score
    from scipy.stats import f


    # Train er function
    @staticmethod
    def nonlinfactor_train_er_func(df_tb3ms: pd.DataFrame,
                                   df_sec_rets: pd.DataFrame,
                                   df_ff_factors: pd.DataFrame,
                                   date_end: int,
                                   forecast_ahead: int = 6,
                                   window_size: int = 36,
                                   func_training_period: int = 36,
                                   hidden_layer_sizes: list = [3],
                                   plot_residual_scatter: bool = False) -> (object, np.array, np.array, np.array):
        '''
        Train the expected return function for the non-linear approach
        Args:
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            date_end: training time window end period
            forecast_ahead: how many periods ahead should the model be trained to forecast
            window_size: return window to use when calculating stock and factor returns.
            func_training_period: how many periods to use to train the nn? func_training_period=1 will use only one cross section, t=date_end
            hidden_layer_sizes: neural net number of units in hidden layer.
            plot_residual_scatter: generate a graph of the residulas for the model
        Returns:
            nn_mod: trained sklearn MLP model object
            X_nlf: X data used to train
            y_train: y data used to train
            y_hat: insample y_hat
        '''

        # Sanity
        if forecast_ahead <= 0:
            raise TypeError('forecast_ahead must be <0 if you are training. Otherwise you will be data snooping...')

        # Prep data for training the nn
        # NB: **the y returned may have invalid stocks missing
        X_nlf, y_nlf = NonLinearFactorInvesting.nonlinfactor_er_func_prep_data(df_tb3ms=df_tb3ms,
                                                      df_sec_rets=df_sec_rets,
                                                      df_ff_factors=df_ff_factors,
                                                      date_end=date_end,  # << Avoid data snooping
                                                      func_training_period=func_training_period,
                                                      forecast_ahead=forecast_ahead,
                                                      window_size=window_size)

        # Normalize X data... and transpose...
        X_norm = StandardScaler().fit_transform(X_nlf)

        # Spilt into train and test data
        # X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y_norm, test_size=0.3, random_state=None)
        X_train_norm = X_norm
        y_train = y_nlf

        # Train ANN... R = f(loadings;factor returns)
        nn_mod = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes),
                              max_iter=1000, # 500
                              learning_rate_init=0.01,
                              random_state=0,
                              solver='adam',  # 'adam', 'sgd'
                              tol=0.001, #= 0.001
                              activation='tanh')  # 'tanh', 'logistic'

        nn_mod = nn_mod.fit(X=X_train_norm, y=np.ravel(y_train))

        # Predict in sample
        y_hat = nn_mod.predict(X_train_norm)
        resid = y_train.values - y_hat

        # Optional ...
        if plot_residual_scatter == True:
            # In sample prediction: Examine residuals for normality...
            # NB: The null hypothesis (H0) states that the variable is normally distributed,
            # and the alternative hypothesis (H1) states that the variable is NOT normally distributed.
            sw_stat, sw_p = shapiro(resid)

            # square plot def
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(y_train.iloc[:, 0] - y_hat, y_hat)
            plt.title('Residual Plot: Shapiro-Wilk p-val: ' + str(round(sw_p, 2)))
            plt.show()

            # Summary stats
            r2 = r2_score(y_true=y_train.values, y_pred=y_hat)
            # F=R2/k(1−R2)/(n−k−1)
            f_stat = r2 / (X_norm.shape[1] * (1 - r2) / (X_norm.shape[0] - X_norm.shape[1] - 1))

            print('R2: ' + str(round(r2, 2)) + '; F-stat: ' + str(round(f_stat, 2)))

        return nn_mod, X_nlf, y_train, y_hat


    # Forecast the expected return of all stocks at a single time point.
    @staticmethod
    def nonlinfactor_forecast_er(nn_model: object,
                                 df_tb3ms: pd.DataFrame,
                                 df_sec_rets: pd.DataFrame,
                                 df_ff_factors: pd.DataFrame,
                                 date_end: int,
                                 window_size: int = 36) -> pd.DataFrame:
        '''
        Forecast expected return for a specific time window ending date_end, using the non-linear approach.
        NB: Forecasts n number of periods into the future is given by the training function
        Args:
            nn_model: MLPRegression object from sklearn
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            date_end: training time window end period
            window_size: return window to use when calculating stock and factor returns.
        Returns:
            e_r: expected returns for all stocks at time point date_end
        '''

        # sanity
        if (window_size < 0) | (window_size > df_sec_rets.shape[0]):
            raise TypeError("(window_size < 0) | (window_size > df_sec_rets.shape[0]")
        if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
            raise TypeError("Must pass factor returns as columns not rows")
        if df_ff_factors.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if (window_size < 0) | (window_size > df_sec_rets.shape[0]):
            raise TypeError("(window_size < 0) | (window_size > df_sec_rets.shape[0]")

        # Factor returns to assume, annualized...
        factor_excess_returns = df_ff_factors.iloc[date_end:, :].sort_index()
        factor_excess_returns = np.array(1 + factor_excess_returns.iloc[date_end:, :]).prod(axis=0)

        # Annulize monthly returns
        if ((window_size) > 12):
            # Supress
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                factor_excess_returns = factor_excess_returns ** (12 / ((window_size))) - 1
        else:
            factor_excess_returns = factor_excess_returns - 1

        # Prep data for training the nn
        # Only use the latets period: func_training_period=1
        # ==================================='
        X_test, y_test = NonLinearFactorInvesting.nonlinfactor_er_func_prep_data(df_tb3ms=df_tb3ms,
                                                        df_sec_rets=df_sec_rets,
                                                        df_ff_factors=df_ff_factors,
                                                        date_end=date_end,
                                                        forecast_ahead=0,  # <<< Set to zero to predict!
                                                        window_size=window_size,
                                                        func_training_period=1)

        X_test_index = X_test.index.to_list()

        # *Quirk ... Sklearn needs 2D array to predict - insert zero row...
        blank_row = np.repeat([999999], X_test.shape[1])
        X_test = np.vstack((X_test, blank_row))

        # Stock Forecast E(R)_i,t+h
        e_r = nn_model.predict(X_test)

        # r_f...
        e_r[(e_r != 0)] = e_r[(e_r != 0)] + df_tb3ms.iloc[date_end, 0]

        # *Remove the added instance (99999) above...
        e_r = pd.DataFrame(e_r, index=None)
        e_r = e_r.iloc[0:-1, :]
        e_r.index = X_test_index

        return e_r


    # Calc all expected returns for all stocks and all time periods.
    # Loop through time, from the earliest period, to the latest, calculating E(R) for every stock we have data for in each time period
    # We will generate a DataFrame containing the expected returns for each stock in each period as we go.
    @staticmethod
    def nonlinfactor_forecast_all_er(df_benchmark_trades: pd.DataFrame,
                                     df_tb3ms: pd.DataFrame,
                                     df_sec_rets: pd.DataFrame,
                                     df_ff_factors: pd.DataFrame,
                                     forecast_ahead: int = 6,
                                     window_size: int = 36,
                                     func_training_period: int = 24) -> (pd.DataFrame, object):
        '''
        Forecast expected return for all time-windows starting at date_start, using the non-linear approach.
        Args:
            df_benchmark_trades: only calculate er for benchmark positions
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            forecast_ahead: how many periods ahead are we forecasting?
            window_size: return window to use when calculating stock and factor returns.
            func_training_period: window to use to train the nn function of loadings and factor returns to expevcted return ...
        Returns:
            df_all_er: expected returns forecast for all stocks across all time periods
        '''

        # E(R) for each stock, in each time period
        df_all_er = pd.DataFrame(np.zeros((df_sec_rets.shape[0] - window_size, df_sec_rets.shape[1])), index=None)
        df_all_er.index = df_sec_rets.index[0:df_sec_rets.shape[0] - window_size]
        df_all_er.columns = df_sec_rets.columns[0:df_sec_rets.shape[1]]  # .astype(int)
        #
        df_stock_SW_pval = df_all_er.copy(deep=True)

        # start period?
        if df_benchmark_trades is None:
            start_period = df_ff_factors.shape[0] - max(func_training_period, window_size) - forecast_ahead
        else:
            start_period = min(df_benchmark_trades.shape[0] - max(func_training_period, window_size) - forecast_ahead, df_ff_factors.shape[0]- max(func_training_period, window_size) - forecast_ahead)

        # Progress
        pbar = tqdm()
        pbar.reset(
            total=start_period - 1)  # initialise with new `total`

        # Step through time... earliest to latest-forecast_ahead.
        for t in range(start_period, -1, -1):

            # Progress
            pbar.update()
            try:
                # Run our function, returning only the result object
                #df_stock_factor_loadings, _, _ = LinearFactorInvesting.factormodel_train_manysecurities(df_tb3ms=df_tb3ms,
                #                                                                  df_sec_rets=df_sec_rets,
                #                                                                  df_ff_factors=df_ff_factors,
                #                                                                  date_start=t + window_size, # << Note we pass in the start date here
                #                                                                  date_end=t,
                #                                                                  test_complexity=False)  # << Note we pass in the end date here

                # Get function of security returns = f(loadings and factor returns)
                nn_mod, _, _, _ = NonLinearFactorInvesting.nonlinfactor_train_er_func(df_tb3ms=df_tb3ms,
                                                             df_sec_rets=df_sec_rets,
                                                             df_ff_factors=df_ff_factors,
                                                             date_end=t,
                                                             forecast_ahead=forecast_ahead,
                                                             window_size=window_size,
                                                             func_training_period=func_training_period)

                # Get forecast returns...
                nlf_er = NonLinearFactorInvesting.nonlinfactor_forecast_er(nn_model=nn_mod,
                                                  df_tb3ms=df_tb3ms,
                                                  df_sec_rets=df_sec_rets,
                                                  df_ff_factors=df_ff_factors,
                                                  date_end=t,
                                                  window_size=window_size)
                er_generated = True
            except:
                if t == start_period:
                    #raise TypeError("No trades created. Cannot continue...")
                    er_generated = False
                else:
                    e_r = pd.DataFrame(nlf_er.iloc[t+1, :].copy(deep=True).T)
                    e_r.columns = ['exp_returns']

            # Forecasts may not have the the cols/stocks of df_sec_rets (invalid cols/stocks will be dropped)
            if er_generated == True:
                row_nlf_er = nlf_er.T
                nlf_er_all_cols = pd.DataFrame(np.zeros((1, df_sec_rets.shape[1])), index=None)
                nlf_er_all_cols.columns = df_sec_rets.columns
                nlf_er_all_cols[row_nlf_er.columns] = row_nlf_er

                # Only keep er values from benchmark, nan all non benchmark stocks
                if df_benchmark_trades is not None:
                    benchmark_mask = df_benchmark_trades.iloc[t, :] == 0 | df_benchmark_trades.iloc[t, :].isna()
                    col_nos_to_nan = list(itertools.compress(range(len(benchmark_mask)), benchmark_mask))
                    cols_to_nan = df_benchmark_trades.columns[col_nos_to_nan]
                    cols_to_nan = [col for col in cols_to_nan if col in e_r.index]
                    e_r.loc[cols_to_nan] = np.nan

                df_all_er.iloc[t, :] = nlf_er_all_cols

                # Set zeros to nan
                def zero_to_nan(x):
                    if x == 0:
                        x = np.nan
                    return x

                df_all_er = df_all_er.applymap(zero_to_nan)

        return df_all_er, nn_mod

class SAIInvesting():
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    @staticmethod
    def sai_er_func_prep_data(df_tb3ms: pd.DataFrame,
                              df_sec_rets: pd.DataFrame,
                              df_ff_factors: pd.DataFrame,
                              dic_fundamentals: dict,
                              date_end: int,
                              func_training_period: int = 1,
                              buysell_threshold_quantile: float = 0,
                              forecast_ahead: int = 6,
                              window_size: int = 36) -> (pd.DataFrame, pd.DataFrame):

        '''
        Prepare data for training, testing and predicting from the SAI model.
        **For training forecast_ahead > 0**
        **For prediction forecast_ahead = 0**
        NB: This look ahead from date_end! Data snooping risk!!
        Note that instances are rejected if 25% or more of the data points are nan
        Args:
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            dic_fundamentals: dictionary containing fundamentals
            date_end: training time window end period
            func_training_period: pass 1 for predictions, >=1 for training. How many periods to use to train the nn? func_training_period=1 will use only one cross section, t=date_end
            buysell_threshold_quantile: which quantile of return should we consider a "buy" and a sell: =0 flags all for rules; positive value flags top/bottom 10th, top/bottom 30th percentile; negative value flags only top
            forecast_ahead: how many periods ahead are we predicting. Set this to 0 if we need data to predict.
            window_size: return window to use when calculating stock and factor returns.
        Returns:
            X: X data used to train/test/predict
            y: X data used to train/test/predict
        '''

        # sanity
        if func_training_period < 1:
            raise TypeError("func_training_period < 1: must have at least 1 training period")
        if df_ff_factors is not None:
            if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
                raise TypeError("Must pass factor returns as columns not rows")
            if df_ff_factors.index[0] != df_sec_rets.index[0]:
                raise TypeError("Dates misaligned")

        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if (func_training_period < 0) | (func_training_period > df_sec_rets.shape[0]):
            raise TypeError("(func_training_period < 0) | (func_training_period > df_sec_rets.shape[0]")
        if (window_size < 0) | (window_size > df_sec_rets.shape[0]):
            raise TypeError("(window_size < 0) | (window_size > df_sec_rets.shape[0]")

        if abs(buysell_threshold_quantile) > 0.5:
            raise TypeError("NB - this function flags the top and bottom using this param: buysell_threshold_quantile > 0.5")

        # Validate: get all dates and tickets from fundamentals
        check_index = None
        check_cols = None
        if dic_fundamentals is not None:
            for fundamentals_df in dic_fundamentals:
                # Indexes should be the same in each df in the dict
                if check_index is None:
                    check_index = list(dic_fundamentals[fundamentals_df].index)
                else:
                    if list(dic_fundamentals[fundamentals_df].index) == check_index:
                        pass
                    else:
                        raise TypeError(
                            'Data mismatch: A df in the fundamentals dictionary has a mismatced index :' + fundamentals_df)

                # Cols should be the same in each df in the dict
                if check_cols is None:
                    check_cols = list(dic_fundamentals[fundamentals_df].columns)
                else:
                    if list(dic_fundamentals[fundamentals_df].columns) != check_cols:
                        raise TypeError(
                            'Data mismatch: A df in the fundamentals dictionary has mismatced columns :' + fundamentals_df)

                # Should have date columns, security rows
                if isinstance(dic_fundamentals[fundamentals_df].columns[0], numbers.Number) == False:
                    raise TypeError('Dict should have date column headers of the form YYYYMM. Getting :' +
                                    dic_fundamentals[fundamentals_df].columns[0])
        else:
            check_index = df_ff_factors.index
            check_cols = df_ff_factors.columns

        all_ticker = list(set(check_index))  # get unique
        all_dates = list(set(check_cols))  # get unique

        # Built training data over their many periods: func_training_period
        # The time points to load are limited by the dtaa we have. Build X and y for window_size, func_training_period, else the longest period available...
        dic_loadings = dict()
        master_X = pd.DataFrame()
        master_y_class = pd.DataFrame()
        master_y_tr = pd.DataFrame()

        for t in range(date_end + forecast_ahead, date_end + forecast_ahead + func_training_period):

            # ================================
            # Get data components for X at time t: df_stock_factor_loadings, factor_excess_returns;
            # and y: stock_returns
            # ================================

            # y variable...
            # =============================
            y_t = pd.DataFrame(np.zeros((len(all_ticker), 0)), index=None)
            # Deduct rf from stock level returns, as it will have been for factor returns
            rf_ret = df_tb3ms.iloc[t - forecast_ahead:t, :].sort_index()
            rf_ret.astype(float)

            # y data... Stock level returns ... forecast ahead by forecast_ahead periods
            stock_returns = df_sec_rets.iloc[t - forecast_ahead:t, :].sort_index()
            stock_returns.astype(float)

            # Adjust returns for r_fs
            for j in range(0, stock_returns.shape[1] - 1):
                stock_returns.iloc[:, j] = (1 + stock_returns.iloc[:, j].values) / (1 + rf_ret.iloc[:, 0].values) - 1  # subtract r_f from each monthly return
            #stock_returns.iloc[:, :] = (1 + stock_returns.iloc[:, :].values) / (1 + rf_ret.iloc[:, 0].values) - 1  # subtract r_f from each monthly return
            stock_returns = np.array(1 + stock_returns).prod(axis=0)

            # Annualize
            if (window_size > 12):
                # Supress warnings here...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stock_returns = stock_returns ** (12 / (window_size)) - 1

            # Add Ticker and Date columns
            y_t = pd.DataFrame(stock_returns, index=None)
            y_t.columns = ['y']

            this_date = df_sec_rets.index[t]
            col_date = [df_sec_rets.index[t] for i in range(0, y_t.shape[0])]
            y_t['date'] = col_date
            y_t['ticker'] = df_sec_rets.columns

            # All dates  and tickers represented?
            # Supress warnings here...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                missing_tickers = [tk for tk in all_ticker if tk not in y_t['ticker'].unique()]
                for missing_ticker in missing_tickers:
                    y_t.loc[y_t.shape[0] + 1] = None
                    y_t.loc[y_t.shape[0], 'ticker'] = missing_ticker
                    y_t.loc[y_t.shape[0], 'date'] = df_sec_rets.index[t]

            # Add X and y at time t to the master X
            y_t['date'] = y_t['date'].astype(int)
            if func_training_period > 1:
                y_t['ticker'] = y_t['ticker'].astype(str) + "_" + str(t)
            else:
                y_t['ticker'] = y_t['ticker'].astype(str)
            y_t = y_t.set_index('ticker')

            # Capture the best and the worst to create SAI rules for...
            # ===========================================================
            # Convert y to a {1,0} classifier {buy, sell}
            if buysell_threshold_quantile == 0:
                # flag all items... with non-zero
                buysell_mask = (y_t['y'] != 0)
            elif buysell_threshold_quantile < 0:
                # flag all items... with non-zero
                buy_threshold = y_t[y_t.isna() == False]['y'].quantile(q=(1-abs(buysell_threshold_quantile)))
                buysell_mask = (y_t['y'] >= buy_threshold)
            else:
                buy_threshold = y_t[y_t.isna() == False]['y'].quantile(q=(1-buysell_threshold_quantile))
                sell_threshold = y_t[y_t.isna() == False]['y'].quantile(q=buysell_threshold_quantile)
                buysell_mask = (y_t['y'] >= buy_threshold) | (y_t['y'] <= sell_threshold)

            # y class...
            y_class_t = y_t.copy(deep=True)
            y_class_t['y'] = 0
            y_class_t.loc[buysell_mask, 'y'] = 1

            # Add to master
            master_y_class = pd.concat((master_y_class, y_class_t), axis=0)
            master_y_tr = pd.concat((master_y_tr, y_t), axis=0)

            # X variable... NO DATA SNOOPING...
            # =============================
            # X Factor Loading data...
            stock_factor_loadings, _, _ = LinearFactorInvesting.factormodel_train_manysecurities(df_tb3ms=df_tb3ms,
                                                                           df_sec_rets=df_sec_rets,
                                                                           df_ff_factors=df_ff_factors,
                                                                           date_start=t + window_size, # << Note we pass in the start date here
                                                                           date_end=t,
                                                                           test_complexity=False)  # << Note we pass in the end date here

            # X...
            # X1) Add fact loadings
            # ---------------------------------------------------
            X_t = pd.DataFrame(np.zeros((stock_factor_loadings.shape[1], stock_factor_loadings.shape[0])), index=None)

            # Add stock loadings to X
            X_t = stock_factor_loadings.T.copy(deep=True)
            X_t.astype(float)
            X_t.columns = stock_factor_loadings.index

            # X2) Add dic_fundamentals entries for the relevant date
            # ---------------------------------------------------
            # Data item, Dataframe per dict element
            # Expecting securities as rows, dates as columns.
            if dic_fundamentals is not None:
                for dict_idx in dic_fundamentals:
                    curr_df = dic_fundamentals[dict_idx]
                    col_mask = curr_df.columns == this_date
                    # Get the date col in this dataitem, set the col name to the di name
                    X_curr_di_t = curr_df.loc[X_t.index, col_mask]

                    # Any data?
                    if X_curr_di_t.shape[1] > 0:
                        X_curr_di_t.columns = [dict_idx]
                        # Non-zero/nulls > 25%?
                        zero_nulls = X_curr_di_t.isna().sum().values + X_curr_di_t[(X_curr_di_t == 0)].sum().values
                        if (zero_nulls / len(X_curr_di_t) < 0.6):
                            # Add to X_t
                            X_t = pd.concat([X_t, X_curr_di_t], axis=1)

            # Refine X
            # X_t = X_t.replace(to_replace=0, value=np.nan)
            # for col_name in X_t.columns:
            #    X_t[col_name] = X_t[col_name].fillna(X_t[col_name].median())

            # Add Ticker and Date columns
            X_t['date'] = col_date
            X_t = X_t.reset_index()
            X_t.columns = X_t.columns.str.lower()

            # All dates  and tickers represented?
            # Supress warnings here...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #
                missing_tickers = [tk for tk in all_ticker if tk not in X_t['ticker'].unique()]
                for missing_ticker in missing_tickers:
                    X_t.loc[X_t.shape[0] + 1] = None
                    X_t.loc[X_t.shape[0], 'ticker'] = missing_ticker
                    X_t.loc[X_t.shape[0], 'date'] = df_sec_rets.index[t]

            # Add X and y at time t to the master X
            X_t['date'] = X_t['date'].astype(int)
            if func_training_period > 1:
                X_t['ticker'] = X_t['ticker'].astype(str) + "_" + str(t)
            else:
                X_t['ticker'] = X_t['ticker'].astype(str)

            X_t = X_t.set_index('ticker')
            master_X = pd.concat((master_X, X_t), axis=0)

        # Final preps...
        # Drop na securities,
        # ======================================
        sai_y_class = master_y_class
        sai_y_tr = master_y_tr
        sai_X = master_X

        # Exclude all nas is we are only using factor loadings.
        # Exclude stocks if there is >25% nan
        if dic_fundamentals is not None:
            X_rows_to_kill = sai_X.loc[sai_X.isna().sum(axis=1) >= 0.25 * sai_X.shape[1]].index.to_list()

        else:
            X_rows_to_kill = sai_X.loc[sai_X.isna().sum(axis=1) != 0].index.to_list()

        y_rows_to_kill = sai_y_class.loc[sai_y_class.isna().sum(axis=1) != 0].index.to_list()
        rows_to_kill = X_rows_to_kill + y_rows_to_kill
        rows_to_kill = list(set(rows_to_kill))
        rows_to_keep = [tk for tk in sai_X.index.to_list() if tk not in X_rows_to_kill]

        # Kill invalid secs from X and y
        sai_X = sai_X.loc[rows_to_keep]
        sai_y_class = sai_y_class.loc[rows_to_keep]
        sai_y_tr = sai_y_tr.loc[rows_to_keep]

        # Drop dates
        sai_y_class = sai_y_class.drop('date', axis=1)
        sai_y_tr = sai_y_tr.drop('date', axis=1)
        sai_X = sai_X.drop('date', axis=1)

        # kill all cols with < 3 unique values
        col_no = 0
        for col_unique_count in sai_X.nunique(axis=0):
            if col_unique_count <= 3:
                sai_X = sai_X.drop(sai_X.columns[col_no])
            col_no += 1

        return sai_X, sai_y_class, sai_y_tr


    import sklearn.metrics as metrics

    # Train er function
    @staticmethod
    def sai_train_er_func(df_tb3ms: pd.DataFrame,
                          df_sec_rets: pd.DataFrame,
                          df_ff_factors: pd.DataFrame,
                          dic_fundamentals: dict,
                          date_end: int,
                          buysell_threshold_quantile: float = 0,
                          lift_cut_off: float = 1,
                          forecast_ahead: int = 6,
                          window_size: int = 36,
                          func_training_period: int = 1,
                          show_analytics: bool = False) -> (object, np.array, np.array, np.array):
        '''
        Args:
          df_tb3ms: risk free rate
          df_sec_rets: stock level returns monthly
          df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
          dic_fundamentals: disctionary containing fundamentals
          date_end: training time window end period
          buysell_threshold_quantile: which quantile of return should we consider a "buy": top 10tb, top 30th percentile
          lift_cut_off: rules must have a lift greater than this cut off (too high and no rules will be selected)
          forecast_ahead: how many periods ahead are we predicting. Set this to 0 if we need data to predict.
          window_size: return window to use when calculating stock and factor returns.
          func_training_period: pass 1 for predictions, >=1 for training. How many periods to use to train the nn? func_training_period=1 will use only one cross section, t=date_end
          plot_residual_analytics: generate a data specific to the model
        Returns:
          sai_mod: trained SAI model object
          X: X data used to train
          y: y data used to train
          y_hat: insample y_hat
        '''

        # sanity
        if not (isinstance(df_tb3ms, pd.DataFrame)):
            raise TypeError("df_tb3ms can only be pandas DataFrame df")
        if not (isinstance(df_sec_rets, pd.DataFrame)):
            raise TypeError("df_sec_rets can only be pandas DataFrame df")
        if not (isinstance(df_ff_factors, pd.DataFrame)):
            raise TypeError("df_ff_factors can only be pandas DataFrame df")
        if dic_fundamentals is not None:
            if not (isinstance(dic_fundamentals, dict)):
                raise TypeError("dic_fundamentals can only be dict")
        if (window_size < 0) | (window_size > df_sec_rets.shape[0]):
            raise TypeError("(window_size < 0) | (window_size > df_sec_rets.shape[0]")

        # ================================
        # train SAI
        # ================================
        # Input parameters
        params = {
            'q': 3,
            'parallel': False,
            'nb_workers': 2,
            'verbose': show_analytics
        }

        # Generate training data
        sai_train_X, sai_train_y_class, sai_train_y_tr = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms,
                                                         df_sec_rets=df_sec_rets,
                                                         df_ff_factors=df_ff_factors,
                                                         dic_fundamentals=dic_fundamentals,
                                                         date_end=date_end+1,  # << Training, No data snooping
                                                         func_training_period=func_training_period,
                                                         buysell_threshold_quantile=buysell_threshold_quantile,
                                                         forecast_ahead=forecast_ahead,
                                                         window_size=window_size)

        # Train SAI...
        # ==========================
        sai_mod = SAI(params=params)
        sai_mod.fit(X=sai_train_X,
                    y=sai_train_y_class,
                    yreal=sai_train_y_tr)

        # Generate test data
        sai_test_X, sai_test_y_class, sai_test_y_tr = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms,
                                                       df_sec_rets=df_sec_rets,
                                                       df_ff_factors=df_ff_factors,
                                                       dic_fundamentals=dic_fundamentals,
                                                       date_end=date_end,  # << Training, No data snooping
                                                       func_training_period=1,
                                                       buysell_threshold_quantile=buysell_threshold_quantile,
                                                       window_size=window_size)  # << Only use 1 period

        # Predict using only rules with lift of > ...
        # ==========================
        y_hat = sai_mod.predict(X=sai_test_X,
                                metric='returns',
                                eval_metric='causal_lift',
                                eval_val=lift_cut_off)

        y_hat = y_hat.sort_index()

        return sai_mod, sai_test_X, sai_test_y_tr, y_hat


    # Forecast er function, requiring a causal_lift>1
    @staticmethod
    def sai_forecast_er(sai_mod: object,
                        df_tb3ms: pd.DataFrame,
                        df_sec_rets: pd.DataFrame,
                        df_ff_factors: pd.DataFrame,
                        dic_fundamentals: dict,
                        date_end: int,
                        buysell_threshold_quantile: float = 0,
                        lift_cut_off: float = 1,
                        window_size: int = 36,
                        training_columns_used: list = None) -> (pd.DataFrame):
        '''
        Args:
          sai_mod: fully ini-ed and trained sai model object
          df_tb3ms: risk free rate
          df_sec_rets: stock level returns monthly
          df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
          dic_fundamentals: disctionary containing fundamentals
          date_end: training time window end period
          buysell_threshold_quantile: which quantile of return should we consider a "buy": top 10tb, top 30th percentile
          lift_cut_off: rules must have a lift greater than this cut off (too high and no rules will be selected)
          window_size: return window to use when calculating stock and factor returns.
          training_columns_used: list the columns of data used to train. A non-None value here will run this function
              with the subset of these columns available in the date_end
        Returns:
          y_hat: insample y_hat
        '''

        # Ini
        func_training_period = 1

        # Get Data
        # ================================
        sai_test_X, _, sai_test_y = SAIInvesting.sai_er_func_prep_data(df_tb3ms=df_tb3ms,
                                                       df_sec_rets=df_sec_rets,
                                                       dic_fundamentals=dic_fundamentals,
                                                       df_ff_factors=df_ff_factors,
                                                       date_end=date_end,  # << Latest period
                                                       func_training_period=1,
                                                       buysell_threshold_quantile=buysell_threshold_quantile,
                                                       forecast_ahead=0,  # << Latest period
                                                       window_size=window_size)  # << Latest period

        # The test data may not contain all the columns in the train data.
        # we can add nan columns for the missing columns....
        if training_columns_used is not None:
            # Add blank column if a column exists in the training data and NOT in the test
            missing_cols_to_add = [col for col in sai_test_X.columns if col not in training_columns_used]
            sai_test_X[[missing_cols_to_add]] = np.nan

        # Predict using only rules with a causal lift of >1
        # ==========================
        y_hat = sai_mod.predict(X=sai_test_X,
                                metric='returns',
                                eval_metric='causal_lift',
                                eval_val=lift_cut_off)
        y_hat = y_hat.sort_index()

        # convert back to dimensions of df_sec_rets
        y_hat_final = pd.DataFrame(np.zeros((df_sec_rets.shape[1], 0)), index=None)
        y_hat_final.index = df_sec_rets.columns
        y_hat_final = y_hat_final.merge(y_hat, how='left', left_index=True, right_index=True)

        return y_hat_final

    # Calc all expected returns for all stocks and all time periods.
    # Loop through time, from the earliest period, to the latest, calculating E(R) for every stock we have data for in each time period
    # We will generate a DataFrame containing the expected returns for each stock in each period as we go.
    @staticmethod
    def sai_forecast_all_er(df_benchmark_trades: pd.DataFrame,
                            df_tb3ms: pd.DataFrame,
                            df_sec_rets: pd.DataFrame,
                            df_ff_factors: pd.DataFrame,
                            dic_fundamentals: dict,
                            date_end: int,
                            buysell_threshold_quantile: float = 0,
                            lift_cut_off: float = 1,
                            forecast_ahead: int = 6,
                            window_size: int = 36,
                            func_training_period: int = 1,
                            plot_residual_analytics: bool = False) -> (pd.DataFrame, object):
        '''
        Args:
            df_benchmark_trades: only calculate er for benchmark positions
            df_tb3ms: risk free rate
            df_sec_rets: stock level returns monthly
            df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series
            dic_fundamentals: disctionary containing fundamentals
            date_end: training time window end period
            buysell_threshold_quantile: which quantile of return should we consider a "buy": top 10tb, top 30th percentile
            forecast_ahead: how many periods ahead are we predicting. Set this to 0 if we need data to predict.
            window_size: return window to use when calculating stock and factor returns.
            func_training_period: pass 1 for predictions, >=1 for training. How many periods to use to train the nn? func_training_period=1 will use only one cross section, t=date_end
            plot_residual_analytics: generate a data specific to the model
        Returns:
            df_all_er: expected returns forecast for all stocks across all time periods
            sai_mod: the last sai model object
        '''

        # E(R) for each stock, in each time period
        df_all_er = pd.DataFrame(np.zeros((df_sec_rets.shape[0] - window_size, df_sec_rets.shape[1])), index=None)
        df_all_er.index = df_sec_rets.index[0:df_sec_rets.shape[0] - window_size]
        df_all_er.columns = df_sec_rets.columns[0:df_sec_rets.shape[1]]  # .astype(int)


        # start period?
        if df_benchmark_trades is None:
            raise TypeError("df_benchmark_trades is required")
        else:
            start_period = min(df_benchmark_trades.shape[0] - forecast_ahead + window_size - func_training_period, df_ff_factors.shape[0] - forecast_ahead - window_size - func_training_period)

        # Progress
        pbar = tqdm()
        pbar.reset(
            start_period - 1)  # initialise with new `total`

        # Step through time... earliest to latest.
        sai_mod = None
        for t in range(start_period, -1, -1):
            # Progress
            pbar.update()

            # Train the SAI model
            try:
                sai_mod, sai_X, _, _ = SAIInvesting.sai_train_er_func(df_tb3ms=df_tb3ms,
                                                         df_sec_rets=df_sec_rets,
                                                         dic_fundamentals=dic_fundamentals,
                                                         df_ff_factors=df_ff_factors,
                                                         date_end=t,  # << Training, No data snooping
                                                         buysell_threshold_quantile=buysell_threshold_quantile,
                                                         lift_cut_off=lift_cut_off,
                                                         forecast_ahead=forecast_ahead,
                                                         window_size=window_size,
                                                         func_training_period=func_training_period)

                # Generate E(R) from our stock level factor model...
                e_r = SAIInvesting.sai_forecast_er(sai_mod=sai_mod,
                                      df_tb3ms=df_tb3ms,
                                      df_sec_rets=df_sec_rets,
                                      df_ff_factors=df_ff_factors,
                                      dic_fundamentals=dic_fundamentals,
                                      date_end=t,
                                      buysell_threshold_quantile=buysell_threshold_quantile,
                                      lift_cut_off=lift_cut_off,
                                      window_size=window_size,
                                      training_columns_used=sai_X.columns)  # << Some data available in the train window may not be available in the test window (or vice versa)

                er_generated = True
            except:
                if t == start_period:
                    #raise TypeError("No trades created. Cannot continue...")
                    er_generated = False
                else:
                    e_r = pd.DataFrame(df_all_er.iloc[t+1, :].copy(deep=True).T)
                    e_r.columns = ['exp_returns']

            # Number of rules?
            if plot_residual_analytics:
                print('Time period: ' + str(t))
                display(sai_mod.rules[(sai_mod.rules['causal_lift']>lift_cut_off)].reset_index(drop=True))

            # Only keep er values from benchmark, nan all non benchmark stocks
            if er_generated == True:
                if df_benchmark_trades is not None:
                    benchmark_mask = df_benchmark_trades.iloc[t, :] == 0 | df_benchmark_trades.iloc[t, :].isna()
                    col_nos_to_nan = list(itertools.compress(range(len(benchmark_mask)), benchmark_mask))
                    cols_to_nan = df_benchmark_trades.columns[col_nos_to_nan]
                    cols_to_nan = [col for col in cols_to_nan if col in e_r.index]
                    e_r.loc[cols_to_nan] = np.nan

                df_all_er.iloc[t, :] = e_r['exp_returns'].T

        return df_all_er, sai_mod

class LinearFactorInvesting():

    import statsmodels.api as sm

    # Forecast the expected return of a single stock
    @staticmethod
    def factormodel_forecast_er(df_stock_factor_loadings: pd.DataFrame,
                                df_ff_factors: pd.DataFrame,
                                r_f: float,
                                date_start: int,
                                date_end: int) -> np.array:

      '''
      Forecast expected return using factor based approach

      Args:
        df_stock_factor_loadings: Factor loadings for all out stocks, and all factors
        df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series, our X variables.
        r_f: Risk free rate to use
        date_start: training time window start period
        date_end: training time window end period

      Returns:
          e_r: expected return forecast

      '''

      # sanity
      if date_start < date_end:
        raise TypeError("Latest date is date=0, date_start is > date_end")
      if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
        raise TypeError("Must pass factor returns as columns not rows")
      if df_ff_factors.shape[1] != df_stock_factor_loadings.shape[0]-1: #Include the intercept dimension
        raise TypeError("Must pass same number of factors for security as the df_ff_factors")

      # Factor returns to assume
      factor_excess_returns = df_ff_factors.iloc[date_end:date_start, :].sort_index()
      factor_excess_returns = np.array(1+factor_excess_returns).prod(axis=0)
      factor_excess_returns = factor_excess_returns ** (12/(date_start-date_end))-1

      # Stock Forecast E(R)_i,t+h
      e_r = np.dot(factor_excess_returns.T, df_stock_factor_loadings.iloc[1:, :])

      #Only add constant and r_f to (non zero returns) populated securities
      non_zero_secs = e_r != 0
      e_r[non_zero_secs] = e_r[non_zero_secs] + r_f + df_stock_factor_loadings.loc['const', non_zero_secs]

      return e_r

    # Function that will run a vectorized OLS model, for a given security, over a given period
    # Vectorized OLS regression is far faster.
    # Note the two optional parameter...
    #   use_robust_cm: estimate from a robust covariance matrix
    #   plot_residual_scatter: which will generate a scatter plot of our residuals (y vs y_hat)
    @staticmethod
    def factormodel_train_manysecurities(df_tb3ms: pd.DataFrame,
                                         df_sec_rets: pd.DataFrame,
                                         df_ff_factors: pd.DataFrame,
                                         date_start: int,
                                         date_end: int,
                                         test_complexity: bool = True) -> (object, np.array, np.array):

        '''
        Calculate all the factor loadings of all our securities
        Args:
          df_tb3ms: Risk free rate timeseries
          df_sec_rets: stock level returns monthly, our y variable
          df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series, our X variables.
          date_start: training time window start period
          date_end: training time window end period
          test_complexity: test model complexity
        Returns:
            ols_model: OLS, sklearn model object
            y: y variable used
            y_hat: in sample forecast of y variable.
        '''

        # sanity
        if date_start < date_end:
            raise TypeError("Latest date is date=0, date_start is > date_end")
        if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
            raise TypeError("Must pass factor returns as columns not rows")
        if df_ff_factors.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")
        if df_tb3ms.index[0] != df_sec_rets.index[0]:
            raise TypeError("Dates misaligned")

            # Get X and y data...
        # NB: Security returns from date_end to date_start... deduct Rf
        y = np.array([df_sec_rets.iloc[t, :] - df_tb3ms.iloc[t, 0] for t in range(date_end, date_start)])
        X = np.array(df_ff_factors.iloc[date_end:date_start, :]).astype('float')

        # Prepare matrices for linalg, and OLS
        intercept = np.ones((date_start - date_end, 1))
        X = np.concatenate((intercept, X), axis=1)

        # Flag nan containing security returns
        cols_non_nan = ~np.isnan(y).any(axis=0)
        y_train = y[:, cols_non_nan]

        # Sanity Check: Biases ************************
        if test_complexity == True:
            failed, _, _ = RobustInvestmentUtils.bias_complexity_check_regression(no_of_instances=X.shape[0],
                                                            # Try to use  36month window to train the MLP
                                                            no_of_features=X.shape[1],  # Do not count intercept
                                                            no_of_parameters=X.shape[1])
            if failed == True:
                print("************ Complexity bias warning ***************")
                # Sanity Check: Biases ************************

        # Train model
        ols_betas, resid, rank, sigma = np.linalg.lstsq(a=X, b=y_train, rcond=None)

        # Predict in sample
        y_hat = np.dot(X, ols_betas)
        resid = y_train - y_hat

        # We removed nan rows... Rebuild to a full vector
        ols_betas_final = pd.DataFrame(np.zeros((ols_betas.shape[0], y.shape[1])), index=None)
        ols_betas_final.loc[:, cols_non_nan] = ols_betas

        ols_betas_final.columns = df_sec_rets.columns
        ols_betas_final.index = ['const'] + df_ff_factors.columns.to_list()
        return (ols_betas_final, y, y_hat)

    # Loop through time, from the earliest period, to the latest, calculating E(R) for every stock we have data for in each time period
    # We will generate a DataFrame containing the expected returns for each stock in each period as we go.
    @staticmethod
    def factormodel_forecast_all_er(df_benchmark_trades: pd.DataFrame,
                                    df_tb3ms: pd.DataFrame,
                                    df_sec_rets: pd.DataFrame,
                                    df_ff_factors: pd.DataFrame,
                                    window_size: int = 36,
                                    factor_return_history: int = 36,
                                    winsorize_er: float = 0) -> pd.DataFrame:

        '''
        Forecast ALL expected returns, for all securities, all time periods, using factor based approach
        Args:
          df_benchmark_trades: only calculate er for benchmark positions, can leave as None...
          df_tb3ms: Risk free rate timeseries
          df_sec_rets: stock level returns monthly, our y variable
          df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series, our X variables.
          window_size: number of months to use, to calculate stock level factor loadings
          factor_return_history: number of months to calculate mean factor returns, which is our assumption for future factor returns
          winsorize_er: Some researchers remove extreme forecasts, specify the higher/lowest winsorize_er percentile of rstock forecasts to remove
        Returns:
            df_stock_er: expected return forecasts
        '''

        # E(R) for each stock, in each time period
        df_stock_er = pd.DataFrame(np.zeros((df_sec_rets.shape[0] - window_size, df_sec_rets.shape[1])), index=None)
        df_stock_er.index = df_sec_rets.index[0:df_sec_rets.shape[0] - window_size]
        df_stock_er.columns = df_sec_rets.columns
        #
        df_stock_SW_pval = df_stock_er.copy(deep=True)

        # start period?
        if df_benchmark_trades is None:
            start_period = df_ff_factors.shape[0] - max(factor_return_history, window_size) -1
        else:
            start_period = min(df_benchmark_trades.shape[0] - max(factor_return_history, window_size), df_ff_factors.shape[0] - max(factor_return_history, window_size))-1

            # Progress
        pbar = tqdm()
        pbar.reset(total=start_period - 1)  # initialise with new `total`

        # Step through time... earliest to latest.
        for t in range(start_period, -1, -1):
            # Progress
            pbar.update()

            # Run our function, returning only the result object
            ols_coefs, _, _ = LinearFactorInvesting.factormodel_train_manysecurities(df_tb3ms=df_tb3ms,
                                                               df_sec_rets=df_sec_rets,
                                                               df_ff_factors=df_ff_factors,
                                                               date_start=t + window_size, # << Note we pass in the start date here
                                                               date_end=t,  # << Note we pass in the end date here
                                                               test_complexity=False)

            # Generate E(R) from our stock level factor model...
            # Factor return assumption
            e_r = LinearFactorInvesting.factormodel_forecast_er(df_stock_factor_loadings=pd.DataFrame(ols_coefs, index=None),
                                          df_ff_factors=df_ff_factors,
                                          r_f=df_tb3ms.iloc[t, 0],
                                          date_start=t + factor_return_history,
                                          date_end=t)


            # Only keep er values from benchmark, nan all non benchmark stocks
            if df_benchmark_trades is not None:
                benchmark_mask = df_benchmark_trades.iloc[t, :] == 0 | df_benchmark_trades.iloc[t, :].isna()
                col_nos_to_nan = list(itertools.compress(range(len(benchmark_mask)), benchmark_mask))
                cols_to_nan = df_benchmark_trades.columns[col_nos_to_nan]
                cols_to_nan = [col for col in cols_to_nan if col in e_r.index]
                e_r.loc[cols_to_nan] = np.nan

            df_stock_er.iloc[t, :] = e_r

        # Set zeros to nan
        def zero_to_nan(x):
            if round(x, 4) == 0:
                x = np.nan
            return x

        df_stock_er = df_stock_er.applymap(zero_to_nan)

        # winzorize?
        if winsorize_er > 0:
            # Get upper and lower limit to remove by col
            upper_kill = df_stock_er.quantile(q=1 - winsorize_er, axis=1)
            lower_kill = df_stock_er.quantile(q=winsorize_er, axis=1)

            def winsorize(row):
                zero_out = (row >= upper_kill[row.name]) | (row <= lower_kill[row.name])
                row.loc[zero_out] = np.nan
                return row

            # Kill df_stock_er
            df_stock_er.apply(winsorize, axis=1)

            # Progress
        pbar.refresh()

        return df_stock_er

    # Function that will run our OLS model to determine factor loadings, for a given security,
    # over a given period
    # Note the two optional parameter...
    #   use_robust_cm: estimate from a robust covariance matrix
    #   plot_residual_scatter: which will generate a scatter plot of our residuals (y vs y_hat)
    def factormodel_train_single_security(sec_col_no: int,
                         df_tb3ms: pd.DataFrame,
                         df_sec_rets: pd.DataFrame,
                         df_ff_factors: pd.DataFrame,
                         date_start: int,
                         date_end: int,
                         use_robust_cm: bool = False,
                         plot_residual_scatter: bool = False) -> (object, np.array, np.array):

      '''
      Calculate the factor loadings of a single security

      Args:
        sec_col_no: Security to select, row number in our dataframe
        df_tb3ms: Risk free rate timeseries
        df_sec_rets: stock level returns monthly, our y variable
        df_ff_factors: DataFrame containing factor return (ie reference portfolio returns such as "value") time series, our X variables.
        date_start: training time window start period
        date_end: training time window end period
        use_robust_cm: use robust standard errors
        plot_residual_scatter: generate a graph of the residulas for the model

      Returns:
          ols_model: OLS, sklearn model object
          y: y variable used
          y_hat: in sample forecast of y variable.

      '''

      # sanity
      if date_start < date_end:
        raise TypeError("Latest date is date=0, date_start is > date_end")
      if df_ff_factors.shape[0] < df_ff_factors.shape[1]:
        raise TypeError("Must pass factor returns as columns not rows")
      if df_ff_factors.index[0] != df_sec_rets.index[0]:
        raise TypeError("Dates misaligned")
      if df_tb3ms.index[0] != df_sec_rets.index[0]:
        raise TypeError("Dates misaligned")

      # Get X and y data...
      # NB: Security returns... deduct Rf
      y = [df_sec_rets.iloc[t, sec_col_no] - df_tb3ms.iloc[t, 0] for t in range(date_end,date_start)]
      X = df_ff_factors.iloc[date_end:date_start, :]

      # Instantiate and train OLS model
      # We will leave the inputs unaltered but if we normalized, it would result in
      # an intercept of aproximately zero, making forecasting down to the stock level betas
      X = sm.add_constant(X) #<< statsmodels requires we manually add an intercept.
      ols_model = OLS(y, X)
      ols_model = ols_model.fit()

      # Optional ... Use heteroskedasticity-autocorrelation robust covariance?
      if use_robust_cm:
          ols_model = ols_model.get_robustcov_results()

      # Predict in sample
      y_hat = ols_model.predict(X)
      resid = y-y_hat

      # Optional ...
      if plot_residual_scatter == True:

        # In sample prediction: Examine residuals for normality...
        # NB: The null hypothesis (H0) states that the variable is normally distributed,
        # and the alternative hypothesis (H1) states that the variable is NOT normally distributed.
        sw_stat, sw_p = shapiro(resid)
        # Check for normality in the residuals
        if sw_p < 0.10:
          print("Residuals appear to be non-normal!") # Make an error in live code

        # square plot
        #plt.clf()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y-y_hat, y_hat)
        plt.title('Residual Plot: Shapiro-Wilk p-val: ' + str(round(sw_p, 2)))
        plt.show()

      return (ols_model, y, y_hat)
