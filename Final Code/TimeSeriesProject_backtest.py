import logging
import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from TimeSeriesProject_toolbox import *
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Login
import wrds
db = wrds.Connection(wrds_username="hanyuzhang")
# password is Timeseries2020!
sym_df = pd.read_csv("Ticker.csv", header=0)
sym_list = sym_df["Ticker"].tolist()
sym_permno_list = list()
for symbol in sym_list:
    if len(symbol) == 0:
        continue
    result = db.raw_sql(f"""select permno, htsymbol 
                           from crsp.dsfhdr 
                           where htsymbol = '{symbol}'""")
    try:
        sym_permno_list.append((result.iloc[0]['permno'], result.iloc[0]['htsymbol']))
    except:
        print(f"wrds doesn't have data for {symbol} right now, skip...")


timeseries_bank = defaultdict(list)
for permno, symbol in sym_permno_list:
    result = db.raw_sql(f"""select date, prc, cfacpr
                            from crsp.dsf
                            where permno = {permno} and date > '2010-01-01'
                         """)
    ts = {date:prc for date, prc in zip(result["date"].tolist(), (result["prc"]/result["cfacpr"]).tolist())}
    if len(ts) != 0:
        timeseries_bank[symbol] = ts

# Build pairs
sym_pairs = list()
sym_pairs_ts = dict()
valid_sym_list = list(timeseries_bank.keys())
for i in range(len(valid_sym_list)-1):
    for j in range(i+1, len(valid_sym_list)):
        sym_pairs.append((valid_sym_list[i], valid_sym_list[j]))
for pair in sym_pairs:
    sym_pairs_ts[pair] = {"first":timeseries_bank[pair[0]], "second":timeseries_bank[pair[1]]}


def cal_spread(first, second, a1, b1) -> float:
    return np.log(first) - a1 - b1 * np.log(second)


def pair_pnl(sym1, sym2, start_date, sigma, mean, half_life, a1, b1, all_pair, trading_cost=0.):
    numoftrade = 0
    numofsucess = 0
    numofdue = 0
    numofstoploss = 0
    numoflowcorr = 0
    numofovertime = 0
    numofwintrade = 0

    open_price = 0
    close_price = 0
    open_value = 0
    close_value = 0
    returns = 0
    growth = 1

    opened = False
    open_date = None
    ls = True
    end_date = start_date + relativedelta(months=3)
    curr_date = start_date
    initial_corr = 0
    sym1_ts = timeseries_bank[sym1]
    sym2_ts = timeseries_bank[sym2]
    dateList = np.array(list(timeseries_bank[sym1].keys())).flatten()
    while curr_date < end_date:
        if curr_date not in sym1_ts or curr_date not in sym2_ts:
            curr_date += relativedelta(days=1)
            continue
        # change
        prev_date_idx = np.where(dateList == curr_date)[0][0] - 1
        prev_date = dateList[prev_date_idx]

        spread = cal_spread(sym1_ts[curr_date], sym2_ts[curr_date], a1, b1)
        if spread > (mean + 2 * sigma) and (end_date - curr_date) > datetime.timedelta(
                days=2 * half_life) and not opened:
            numoftrade += 1
            opened = True
            ls = True
            open_date = curr_date
            open_price = -sym1_ts[curr_date] * (1.0 - trading_cost) + b1 * sym2_ts[curr_date] * (1.0 + trading_cost)
            open_value = b1 * sym2_ts[curr_date] * (1.0 + trading_cost)
            close_value = sym1_ts[curr_date] * (1.0 - trading_cost)
            initial_corr = \
            CorrelationTest(all_pair, curr_date - relativedelta(years=1), curr_date, timeseries_bank, 5, func1)[
                (sym1, sym2)]
            logger.debug(f"Open pair trade S{sym1, sym2}L on {curr_date}, price at open is {open_price}")
        elif spread < (mean - 2 * sigma) and (end_date - curr_date) > datetime.timedelta(
                days=2 * half_life) and not opened:
            numoftrade += 1
            opened = True
            ls = False
            open_date = curr_date
            open_price = sym1_ts[curr_date] * (1.0 + trading_cost) - b1 * sym2_ts[curr_date] * (1.0 - trading_cost)
            open_value = sym1_ts[curr_date] * (1.0 + trading_cost)
            close_value = b1 * sym2_ts[curr_date] * (1.0 - trading_cost)
            initial_corr = \
            CorrelationTest(all_pair, curr_date - relativedelta(years=1), curr_date, timeseries_bank, 5, func1)[
                (sym1, sym2)]
            logger.debug(f"Open pair trade L{sym1, sym2}S on {curr_date}, price at open is {open_price}")
        # trade close
        if opened:
            curr_corr = \
            CorrelationTest(all_pair, curr_date - relativedelta(years=1), curr_date, timeseries_bank, 5, func1)[
                (sym1, sym2)]
            if (((spread <= mean + 2 * sigma) and (spread >= mean - 2 * sigma)) or
                    (curr_date - open_date) > datetime.timedelta(days=10 * int(half_life)) or
                    (curr_corr < initial_corr * 0.8) or
                    ((spread >= mean + 3 * sigma) and (spread <= mean - 3 * sigma))):
                if ((spread <= mean + 2 * sigma) and (spread >= mean - 2 * sigma)):
                    numofsucess += 1
                    #print((sym1, sym2), "Sucessfully close")
                elif ((spread >= mean + 3 * sigma) and (spread <= mean - 3 * sigma)):
                    numofstoploss += 1
                    #print((sym1, sym2), "fail close,stop loss here")
                elif (curr_date - open_date) > datetime.timedelta(days=10 * int(half_life)):
                    numofovertime += 1
                    #print((sym1, sym2), "fail close, reach 3*half-life")
                else:
                    #print((sym1, sym2), "fail close, lower correlation")
                    numoflowcorr += 1

                if ls:
                    close_price = -sym1_ts[curr_date] * (1 + trading_cost) + b1 * sym2_ts[curr_date] * (
                                1 - trading_cost)
                    close_value += (
                                b1 * sym2_ts[curr_date] * (1 - trading_cost) - sym1_ts[curr_date] * (1 + trading_cost))
                    growth *= (close_value / open_value)
                    returns += (close_price - open_price)
                    if (close_price - open_price) > 0:
                        numofwintrade += 1
                    logger.debug(f"Close pair trade S{sym1, sym2}L on {curr_date}, price at close is {close_price}")
                else:
                    close_price = sym1_ts[curr_date] * (1 - trading_cost) - b1 * sym2_ts[curr_date] * (1 + trading_cost)
                    close_value += (
                                sym1_ts[curr_date] * (1 - trading_cost) - b1 * sym2_ts[curr_date] * (1 + trading_cost))
                    growth *= (close_value / open_value)
                    returns += (close_price - open_price)
                    if (close_price - open_price) > 0:
                        numofwintrade += 1
                    logger.debug(f"Close pair trade L{sym1, sym2}S on {curr_date}, price at close is {close_price}")
                opened = False
                break
        curr_date += relativedelta(days=1)
    if opened:
        #print((sym1, sym2), "fail close,time is due")
        numofdue += 1
        curr_date -= relativedelta(days=1)
        while curr_date not in sym1_ts:
            curr_date -= relativedelta(days=1)
        if ls:
            close_price = (-sym1_ts[curr_date] + b1 * sym2_ts[curr_date]) * (1 - trading_cost)
            returns += (close_price - open_price)
            growth *= (close_value / open_value)
            if (close_price - open_price) > 0:
                numofwintrade += 1
        else:
            close_price = (sym1_ts[curr_date] - b1 * sym2_ts[curr_date]) * (1 - trading_cost)
            returns += (close_price - open_price)
            growth *= (close_value / open_value)
            if (close_price - open_price) > 0:
                numofwintrade += 1
        logger.debug(f"Close pair trade {sym1, sym2} on {curr_date}, price at close is {close_price}")
        opened = False
    return returns, growth - 1, {"TotalTrade": numoftrade, "Success": numofsucess, "Due": numofdue,
                                 "Halflife": numofovertime,
                                 "Corr": numoflowcorr, "Stoploss": numofstoploss, "WinTrade": numofwintrade}


def pnl_wrapper(quarter, verbose=False, corr_bench=0.0, trading_cost=0.0):
    return_summary = {}
    return_rate_list = []
    quarterly_stats = {"TotalTrade": 0, "Success": 0, "Due": 0, "Halflife": 0,
                       "Corr": 0, "Stoploss": 0, "WinTrade": 0}
    for pair in sym_pairs:
        return_summary[pair] = 0.
    total_pnl = 0
    end_date = quarter - relativedelta(days=1)
    start_date = quarter - relativedelta(years=1)
    # pair selection
    post_coint_pairs = CointegrationTest(sym_pairs, start_date, end_date, timeseries_bank)
    pair_corr = CorrelationTest(post_coint_pairs, start_date, end_date, timeseries_bank, 5, func2)
    post_corr_pairs = list()
    for pair, corr in pair_corr.items():
        if corr > corr_bench:
            post_corr_pairs.append(pair)
    if len(post_corr_pairs) > 0:
        pair_params = OUCalibration(post_corr_pairs, start_date, end_date, sym_pairs_ts)
        for pair, params in pair_params.items():
            pair_result, return_rate, stats = pair_pnl(pair[0], pair[1], quarter, params["sigma"], params["mu"],
                                                       params["half_life"], params["const"], params["coef"],
                                                       post_corr_pairs, trading_cost)
            for key, value in stats.items():
                quarterly_stats[key] = quarterly_stats[key] + value

            print(pair, "quarterly earn:", pair_result)
            return_rate_list.append(return_rate)
            total_pnl += pair_result
            return_summary[pair] = pair_result

    logger.info(f"total pnl for quarter {quarter} is {total_pnl}")

    return total_pnl, return_summary, quarterly_stats, return_rate_list


def full_test(start_date, corr_bench=0.0, trading_cost=0.0, verbose=False):
    import matplotlib.pyplot as plt
    summary = {}
    stats = []
    total_return_rate = []
    date_track = []
    for pair in sym_pairs:
        summary[pair] = 0
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    quarters = list()
    pnls = list()
    curr_date = start_date
    while curr_date < datetime.date(2019, 12, 31):
        pnl, Rsummary, quarter_stats, quarter_return_rate = pnl_wrapper(curr_date, corr_bench=corr_bench,
                                                                        trading_cost=trading_cost)
        total_return_rate = total_return_rate + quarter_return_rate
        pnls.append(pnl)
        date_track.append(curr_date)
        quarters.append(curr_date)
        curr_date += relativedelta(months=3)
        for key, value in Rsummary.items():
            summary[key] = summary[key] + value
        stats.append(list(quarter_stats.values()))


    return pnls, summary, stats, np.array(total_return_rate), date_track

#run back-test Now!
#pnls,summary,stats=full_test(datetime.date(2011, 1, 10))