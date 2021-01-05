import numpy as np
import pandas as pd
import wrds
from collections import defaultdict
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import datetime
import statsmodels.api as sm

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def CointegrationTest(pairs, start_date, end_date, data):
    """
    pair: list of tuple
    date: datetime.date(YYYY,MM,DD)
    data: timeseries_bank
    """
    final_pairs = []
    for pair in pairs:
        timestamp = np.array(list(data[pair[0]].keys())).flatten()
        # print(pair,len(timestamp),len(list(data[pair[0]].values())),len(list(data[pair[1]].values())))
        t1 = np.log(
            np.array(list(data[pair[0]].values())).flatten()[(timestamp >= start_date) & (timestamp <= end_date)])
        t2 = np.log(
            np.array(list(data[pair[1]].values())).flatten()[(timestamp >= start_date) & (timestamp <= end_date)])

        t1 = t1 / t1[0]
        t2 = t2 / t2[0]

        result = coint_johansen(np.array([t1, t2]).T, det_order=0, k_ar_diff=1)
        if np.all(result.lr1 >= result.trace_stat_crit_vals[:, 0]):
            final_pairs.append(pair)
    return final_pairs


# functions for calcualting estimiated correlation matrix, inout of method "CorrelationTest"
func1=lambda w,V,numOfPC:V[:,:numOfPC].dot(np.diag(w[:numOfPC])).dot(V[:,:numOfPC].T)
func2=lambda w,V,numOfPC:V[:,:numOfPC].dot(np.diag(w[:numOfPC])).dot(V[:,:numOfPC].T)+\
                          sum(w[numOfPC:])/(len(w)-numOfPC)*(V[:,numOfPC:].dot(V[:,numOfPC:].T))


def CorrelationTest(pairs, start_date, end_date, data, numOfPC, func):
    """
    pair: list of tuple
    date: datetime.date(YYYY,MM,DD)
    data: timeseries_bank
    numOfCP: number of PC from evaluating correlation, we better has numOfPC<len(pairs)/2
    func: ways to calculate reduced corr
    """
    tickerList = []
    for pair in pairs:
        tickerList = tickerList + list(pair)
    tickerList = list(set(tickerList))

    raw_data = pd.DataFrame()
    timestamp = np.array(list(data[tickerList[0]].keys())).flatten()
    dateRange = (timestamp >= start_date) & (timestamp <= end_date)
    for ticker in tickerList:
        temp = np.array(list(data[ticker].values())).flatten()[dateRange]
        raw_data[ticker] = pd.Series(temp)
    return_data = np.log(1 + raw_data.pct_change().dropna())
    w, V = np.linalg.eig((return_data - return_data.mean()).cov())
    new_cov = func(w, V, numOfPC)
    new_corr = pd.DataFrame(correlation_from_covariance(new_cov), index=tickerList, columns=tickerList)
    ans = {}
    for pair in pairs:
        ans[pair] = new_corr.loc[pair[0], pair[1]]
    return ans


def OUCalibration(pairs, start_date, end_date, data):
    """
    pair: list of tuple
    date: datetime.date(YYYY,MM,DD)
    data: sym_pairs_ts
    """
    timestamp = np.array(list(data[pairs[0]]["first"].keys())).flatten()
    dateRange = (timestamp >= start_date) & (timestamp <= end_date)
    ans = {}
    deltat = 1 / 252
    for pair in pairs:
        t1 = np.log(np.array(list(data[pair]["first"].values())).flatten()[dateRange])
        t2 = np.log(np.array(list(data[pair]["second"].values())).flatten()[dateRange])

        t2_constant = sm.add_constant(t2)
        model2 = sm.OLS(t1, t2_constant)
        result2 = model2.fit()
        a1, b1 = result2.params
        #print(np.mean(result2.resid))

        Xt = t1 - a1 - b1 * t2
        y = Xt[1:]
        x = Xt[:-1]
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        a, b = results.params
        std_residual = np.std(results.resid)

        theta = -np.log(b) / deltat
        mu = a / (1 - b)
        sigma = std_residual * np.sqrt(2 * theta / (1 - b ** 2))
        #print(mu)

        ans[pair] = {"half_life": np.log(2) / theta * 252, "mu": mu, "sigma": sigma / np.sqrt(2 * theta), "const": a1,
                     "coef": b1}

    return ans



if __name__=="__main__":
    db = wrds.Connection(wrds_username="hanyuzhang")

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
        ts = {date: prc for date, prc in zip(result["date"].tolist(), (result["prc"] / result["cfacpr"]).tolist())}
        timeseries_bank[symbol] = ts

    # Build pairs
    sym_pairs = list()
    sym_pairs_ts = dict()
    valid_sym_list = list(timeseries_bank.keys())
    for i in range(len(valid_sym_list) - 1):
        for j in range(i + 1, len(valid_sym_list)):
            sym_pairs.append((valid_sym_list[i], valid_sym_list[j]))
    for pair in sym_pairs:
        sym_pairs_ts[pair] = {"first": timeseries_bank[pair[0]], "second": timeseries_bank[pair[1]]}

    listOfPairs = list(sym_pairs_ts.keys())

    startDate = datetime.date(2012, 10, 1)
    endDate = datetime.date(2013, 10, 1)
    finalPair = CointegrationTest(listOfPairs, startDate, endDate, timeseries_bank)
    print(finalPair)
    correlation1 = CorrelationTest(listOfPairs, startDate, endDate, timeseries_bank, 8, func1)
    print(correlation1)
    correlation2 = CorrelationTest(listOfPairs, startDate, endDate, timeseries_bank, 8, func2)
    print(correlation2)
