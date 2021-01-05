import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pnl(pnls):
    plt.plot(np.cumsum(pnls))
    plt.show()


def profit_attribution(ticker,summary):
    df=pd.DataFrame(columns=ticker,index=ticker)
    for t1 in ticker:
        for t2 in ticker:
            if t1 != t2:
                try:
                    df.loc[t1, t2] = float(summary[(t1, t2)])
                    df.loc[t2, t1] = float(summary[(t1, t2)])
                except:
                    try:
                        df.loc[t1, t2] = float(summary[(t2, t1)])
                        df.loc[t2, t1] = float(summary[(t2, t1)])
                    except:
                        raise ValueError("error")
    f = plt.figure(figsize=(16, 8))
    plt.matshow(df.astype(float), fignum=f.number)
    plt.xticks(range(len(ticker)), ticker, fontsize=14, rotation=45)
    plt.yticks(range(len(ticker)), ticker, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Total Profit Attribution', fontsize=16)
    plt.show()

def show_winning_rate(stats,date_track):
    stats=np.array(stats)
    df = pd.DataFrame(
        {"Total Trade": stats[:, 0], "Winning Trade": stats[:, -1], "winning rate": stats[:, -1] / stats[:, 0]},
        index=date_track)
    df.fillna(0, inplace=True)

    plt.figure(figsize=(15, 10))
    width = 0.3
    ind = np.arange(len(date_track))
    plt.bar(ind, df["Total Trade"], width, color='orange', label="Total Trade")
    plt.bar(ind + width, df["Winning Trade"], width, color='g', label="Winning Trade")
    plt.ylabel('# of trades', fontsize=18)
    plt.legend()
    axes2 = plt.twinx()
    axes2.plot(ind, df["winning rate"], color='b', label='winning rate')
    axes2.set_ylim(0, 1.01)
    axes2.set_ylabel('winning rate', fontsize=18)
    plt.legend()

    plt.xticks(ind, date_track, fontsize=18)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel("time")
    plt.show()

def show_success_rate(stats,date_track):
    stats=np.array(stats)
    df = pd.DataFrame(
        {"Total Trade": stats[:, 0], "Success Trade": stats[:, 1], "successful close rate": stats[:, 1] / stats[:, 0]},
        index=date_track)
    df.fillna(0, inplace=True)

    plt.figure(figsize=(15, 10))
    width = 0.3
    ind = np.arange(len(date_track))
    plt.bar(ind, df["Total Trade"], width, color='orange', label="Total Trade")
    plt.bar(ind + width, df["Success Trade"], width, color='g', label="Success Trade")
    plt.ylabel('# of trades', fontsize=18)
    plt.legend()
    axes2 = plt.twinx()
    axes2.plot(ind, df["successful close rate"], color='b', label='successful close rate')
    axes2.set_ylim(0, 1.01)
    axes2.set_ylabel('successful close rate', fontsize=18)
    plt.legend()

    plt.xticks(ind, date_track, fontsize=18)
    plt.locator_params(axis='x', nbins=10)
    plt.xlabel("time")
    plt.show()

def get_trade_table(stats,date_track):
    df = pd.DataFrame(stats)
    df.index = date_track
    df.columns = ["TotalTrade", "Success Trade", " Closed Trade - time due", " Closed Trade - Halflife",
                  " Closed Trade - lower corr", " Closed Trade - stop loss", "Winning Trade"]
    return df[["TotalTrade","Winning Trade","Success Trade",  " Closed Trade - time due",
                  " Closed Trade - Halflife"," Closed Trade - lower corr"," Closed Trade - stop loss"]]

def stats_table(rate):
    df = pd.DataFrame(rate)
    sumdf = pd.DataFrame()
    for a in [1, 0.99, 0.95, 0.9, 0.8]:
        mean, std = df[(df < df.quantile(a)) & (df > df.quantile(1 - a))].describe().loc[
            ["mean", 'std']].values.flatten()
        b = np.round(1 - a, 2)
        string = " {}<= quantile <= {}".format(b, a)
        mean = mean * 4
        std = std * np.sqrt(4)
        sumdf[string] = [mean, std, (mean - 0.0005) / std]

    sumdf.index = ["mean", "std", "Sharpe ratio"]

    return sumdf