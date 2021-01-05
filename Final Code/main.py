from TimeSeriesProject_backtest import *
from TimeSeriesProject_analysis import *

#back_test
pnls,summary,stats,rates,datetrack=full_test(datetime.date(2011, 1, 10),corr_bench=0.0,trading_cost=0.0)

#analysis
print(stats)
ticker=list(timeseries_bank.keys())
plot_pnl(pnls)
profit_attribution(ticker,summary)
show_success_rate(stats,datetrack)
show_winning_rate(stats,datetrack)
trade_table=get_trade_table(stats,datetrack)
trade_table.to_csv("trade_table.csv")
print(trade_table)
rate_summary=stats_table(rates)
rate_summary.to_csv("rate_summary.csv")
print(rate_summary)