import pandas as pd
from pandas import DataFrame, Series
from typing import TypedDict, List, Dict, Any
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import importlib.util
import sys
import os
from importlib.machinery import ModuleSpec
from types import ModuleType, FunctionType
from matplotlib.axes import Axes
import json

# CONSTANTS #######################################################################################
START_DAY: int = 0
END_DAY: int = 0
INSTRUMENT_POSITION_LIMIT: int = 10000
COMMISSION_RATE: float = 0.0005
NUMBER_OF_INSTRUMENTS: int = 50

PLOT_COLORS: Dict[str, str] = {
    "pnl": "#2ca02c",
    "cum_pnl": "#1f77b4",
    "utilisation": "#ff7f0e",
    "sharpe_change": "#d62728",
}

default_strategy_filepath: str = "./main.py"
default_strategy_function_name: str = "getMyPosition"
strategy_file_not_found_message: str = "Strategy file not found"
could_not_load_spec_message: str = "Could not load spec for module from strategy file"
strategy_function_does_not_exist_message: str = (
    "getMyPosition function does not exist in strategy file"
)
strategy_function_not_callable_message: str = "getMyPosition function is not callable"

usage_error: str = """
    Usage: backtester.py [OPTIONS]

    OPTIONS: You can only specify each option once 
    --path [filepath: string] supply a custom filepath to your .py file that holds your
        getMyPosition() function. If not specified, it will use the filepath "./main.py"
    --function-name [function_name: string] supply a custom 'getMyPositions' function name.
        this function must take an 2-dimensional ndarray with a length of 50 and return
        an ndarray of length 50 that represent positions for each instruments
    --timeline [start_day: int] [end_day: int] supply a custom start day and end day to run the
        backtester in. start day >= 1 and end day <= 1000. If not specified, backtester will run
        throughout days 1-1000
    --disable-comms disable commission on trades
    --show [graph1 graph2 ...] - specify which graphs to show. If this option is not specified, 
        by default the backtester will show cumulative PnL, daily PnL and capital utilisation. A 
        max of 3 graphs can be specified

        Available graphs:
            daily-pnl: graphs your daily profit and loss
            cum-pnl: graphs your cumulative profit and loss over time
            capital-util: graphs your total capital utilisation over time
            sharpe-heat-map: graphs a heat map of the daily sharpe ratios your strategy had over 
                time
            cum-sharpe: graphs your cumulative sharpe ratio over time
"""

CMD_LINE_OPTIONS: List[str] = [
    "--path",
    "--function-name",
    "--timeline",
    "--disable-comms",
    "--show",
]

GRAPH_OPTIONS: List[str] = [
    "daily-pnl",
    "cum-pnl",
    "capital-util",
    "sharpe-heat-map",
    "cum-sharpe",
]

# TYPE DECLARATIONS ###############################################################################
class InstrumentPriceEntry(TypedDict):
    day: int
    instrument: int
    price: float

class Trade(TypedDict):
    price_entry: float
    order_type: str
    day: int

class BacktesterResults(TypedDict):
    daily_pnl: ndarray
    daily_capital_utilisation: ndarray
    daily_instrument_returns: ndarray
    trades: Dict[int, List[Trade]]
    start_day: int
    end_day: int

class Params:
    def __init__(
            self,
            strategy_filepath: str = default_strategy_filepath,
            strategy_function_name: str = default_strategy_function_name,
            strategy_function: FunctionType | None = None,
            start_day: int = 1,
            end_day: int = 1000,
            enable_commission: bool = True,
            graphs: List[str] = ["cum-pnl", "sharpe-heat-map", "daily-pnl"],
            prices_filepath: str = "./prices.txt",
            instruments_to_test: List[int] = range(1, 51)
    ) -> None:
        self.strategy_filepath = strategy_filepath
        self.strategy_function_name = strategy_function_name
        self.strategy_function = strategy_function
        self.start_day = start_day
        self.end_day = end_day
        self.enable_commission = enable_commission
        self.graphs = graphs
        self.prices_filepath: str = prices_filepath
        self.instruments_to_test: List[int] = instruments_to_test

# HELPER FUNCTIONS ###############################################################################
def parse_command_line_args() -> Params:
    total_args: int = len(sys.argv)
    params: Params = Params()

    if total_args > 1:
        i: int = 1
        while i < total_args:
            current_arg: str = sys.argv[i]

            if current_arg == "--path":
                if i + 1 >= total_args:
                    raise Exception(usage_error)
                else:
                    i += 1
                    params.strategy_filepath = sys.argv[i]
            elif current_arg == "--timeline":
                if i + 2 >= total_args:
                    raise Exception(usage_error)
                else:
                    params.start_day = int(sys.argv[i + 1])
                    params.end_day = int(sys.argv[i + 2])
                    i += 2

                    if (
                            params.start_day > params.end_day
                            or params.start_day < 1
                            or params.end_day > 1000
                    ):
                        raise Exception(usage_error)
            elif current_arg == "--disable-comms":
                params.enable_commission = False
            elif current_arg == "--function-name":
                if i + 1 >= total_args:
                    raise Exception(usage_error)
                else:
                    params.strategy_function_name = sys.argv[i + 1]
                    i += 1
            elif current_arg == "--show":
                if i + 1 >= total_args:
                    raise Exception(usage_error)

                params.graphs = []
                i += 1
                current_arg = sys.argv[i]
                while current_arg not in CMD_LINE_OPTIONS:
                    if current_arg not in GRAPH_OPTIONS or len(params.graphs) == 3:
                        raise Exception(usage_error)

                    params.graphs.append(current_arg)
                    i += 1
                    if i < total_args:
                        current_arg = sys.argv[i]
                    else:
                        break
                i -= 1
            else:
                raise Exception(usage_error)

            i += 1

    return params

def load_get_positions_function(
        strategy_filepath: str, strategy_function_name: str
) -> FunctionType:
    filepath: str = os.path.abspath(strategy_filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(strategy_file_not_found_message)
    module_name: str = os.path.splitext(os.path.basename(filepath))[0]
    spec: ModuleSpec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        raise ImportError(could_not_load_spec_message)
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, strategy_function_name):
        raise AttributeError(strategy_function_does_not_exist_message)
    function = getattr(module, strategy_function_name)
    if not callable(function):
        raise TypeError(strategy_function_not_callable_message)
    return function

def generate_stats_subplot(
        results: BacktesterResults, subplot: Axes, enable_commission: bool
) -> Axes:
    subplot.axis("off")
    win_rate_pct: float = (
            np.sum(results["daily_pnl"] > 0) / len(results["daily_pnl"]) * 100
    )
    stats_text: str = (
            f"Ran from day {results['start_day']} to {results['end_day']}\n"
            r"$\bf{Commission \ Turned \ On:}$" + f"{enable_commission}\n\n"
            r"$\bf{Backtester \ Stats}$" + "\n\n"
            f"Mean PnL: ${results['daily_pnl'].mean():.2f}\n"
            f"Std Dev: ${results['daily_pnl'].std():.2f}\n"
            f"Annualised Sharpe Ratio: "
            f"{np.sqrt(250) * results['daily_pnl'].mean() / results['daily_pnl'].std():.2f}\n"
            f"Win Rate %: {win_rate_pct:.2f}% \n"
            f"Score: {results['daily_pnl'].mean() - 0.1 * results['daily_pnl'].std():.2f}"
    )
    subplot.text(
        0.05, 0.95, stats_text, fontsize=14, va="top", ha="left", linespacing=1.5
    )
    return subplot

def generate_cumulative_pnl_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    cumulative_pnl: ndarray = np.cumsum(results["daily_pnl"])
    days: ndarray = np.arange(results["start_day"], results["end_day"] + 1)
    subplot.set_title(
        f"Cumulative Profit and Loss from day {results['start_day']} to {results['end_day']}",
        fontsize=12, fontweight="bold"
    )
    subplot.set_xlabel("Days", fontsize=10)
    subplot.set_ylabel("Total PnL ($)", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.plot(days, cumulative_pnl, linestyle="-", color=PLOT_COLORS["cum_pnl"], linewidth=2)
    return subplot

def generate_daily_pnl_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    days: ndarray = np.arange(results["start_day"], results["end_day"] + 1)
    subplot.set_title(
        f"Daily Profit and Loss (PnL) from day {results['start_day']} to {results['end_day']}",
        fontsize=12, fontweight="bold"
    )
    subplot.set_xlabel("Days", fontsize=10)
    subplot.set_ylabel("PnL ($)", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.plot(days, results["daily_pnl"], linestyle="-", color=PLOT_COLORS["pnl"])
    return subplot

def generate_capital_utilisation_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    daily_capital_utilisation_pct: ndarray = results["daily_capital_utilisation"] * 100
    days: ndarray = np.arange(results["start_day"], results["end_day"] + 1)
    subplot.set_title(
        f"Daily capital utilisation from day {results['start_day']} to {results['end_day']}",
        fontsize=12, fontweight="bold"
    )
    subplot.set_xlabel("Days", fontsize=10)
    subplot.set_ylabel("Capital Utilisation %", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.set_ylim(0, 100)
    subplot.plot(days, daily_capital_utilisation_pct, linestyle="-", color=PLOT_COLORS["utilisation"])
    return subplot

def generate_sharpe_heat_map(results: BacktesterResults, subplot: Axes) -> Axes:
    returns: ndarray = results["daily_instrument_returns"]
    means: ndarray = np.mean(returns, axis=1)
    stds: ndarray = np.std(returns, axis=1)
    sharpe_ratios: ndarray = (means / stds) * np.sqrt(250)
    sharpe_grid = sharpe_ratios.reshape(1, -1)
    im = subplot.imshow(sharpe_grid, cmap="viridis", aspect="auto")
    subplot.set_title("Annualised Sharpe-Ratio Heat Map (Higher = Better)", fontsize=12)
    subplot.set_xticks(np.arange(len(sharpe_ratios)))
    subplot.set_xticklabels([str(i) for i in range(len(sharpe_ratios))], fontsize=6)
    subplot.set_yticks([])
    color_bar = subplot.figure.colorbar(im, ax=subplot, orientation="vertical", pad=0.01)
    color_bar.set_label("Sharpe", fontsize=9)
    return subplot

def generate_sharpe_ratio_subplot(results: BacktesterResults, subplot: Axes) -> Axes:
    daily_pnl: ndarray = results["daily_pnl"]
    counts: ndarray = np.arange(1, results["end_day"] - results["start_day"] + 2)
    days: ndarray = np.arange(results["start_day"], results["end_day"] + 1)
    cumulative_pnl: ndarray = np.cumsum(daily_pnl)
    cumulative_means: ndarray = cumulative_pnl / counts
    cumulative_std_dev: ndarray = np.array(
        [np.std(daily_pnl[: i + 1], ddof=0) for i in range(len(daily_pnl))]
    )
    sharpe_ratios: ndarray = (cumulative_means / cumulative_std_dev) * np.sqrt(250)
    subplot.set_title(
        f"Change in Annualised Sharpe Ratio from day {results['start_day']} to {results['end_day']}",
        fontsize=12, fontweight="bold"
    )
    subplot.set_xlabel("Days", fontsize=10)
    subplot.set_ylabel("Annualised Sharpe Ratio", fontsize=10)
    subplot.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.plot(days, sharpe_ratios, linestyle="-", color=PLOT_COLORS["sharpe_change"])
    return subplot

def get_subplot(graph_type: str, results: BacktesterResults, subplot: Axes) -> Axes:
    if graph_type == "daily-pnl":
        return generate_daily_pnl_subplot(results, subplot)
    elif graph_type == "cum-pnl":
        return generate_cumulative_pnl_subplot(results, subplot)
    elif graph_type == "capital-util":
        return generate_capital_utilisation_subplot(results, subplot)
    elif graph_type == "sharpe-heat-map":
        return generate_sharpe_heat_map(results, subplot)
    elif graph_type == "cum-sharpe":
        return generate_sharpe_ratio_subplot(results, subplot)

def get_ema(instrument_price_history: ndarray, lookback: int) -> ndarray:
    price_series: Series = pd.Series(instrument_price_history)
    return price_series.ewm(span=lookback, adjust=False).mean()

# BACKTESTER CLASS ################################################################################
class Backtester:
    def __init__(self, params: Params) -> None:
        self.enable_commission: bool = params.enable_commission
        self.getMyPosition: FunctionType | None
        if params.strategy_function is not None:
            self.getMyPosition = params.strategy_function
        else:
            self.getMyPosition = load_get_positions_function(
                params.strategy_filepath, params.strategy_function_name
            )
        self.raw_prices_df: DataFrame = pd.read_csv(
            params.prices_filepath, sep=r"\s+", header=None, engine='python'
        )
        self.price_history: ndarray = self.raw_prices_df.to_numpy().T

    def run(
            self,
            start_day: int,
            end_day: int,
            config: Dict[int, Dict[str, Dict[str, float]]] | None = None,
            instruments_to_test: List[int] | None = None
    ) -> BacktesterResults:
        current_positions: ndarray = np.zeros(NUMBER_OF_INSTRUMENTS)
        cash: float = 0
        portfolio_value: float = 0
        position_history: Dict[int, list[int]] = {instrument: [0] for instrument in range(0, 50)}
        daily_pnl_list: List[float] = []
        daily_capital_utilisation_list: List[float] = []
        instrument_returns: Dict[int, list[float]] = {instrument: [0] for instrument in range(0, 50)}
        trades: Dict[int, List[Trade]] = {instrument: [] for instrument in range(0, 50)}
        requested_positions_history: List[List[int]] = []
        for instrument_no in range(0, 50):
            requested_positions_history.append([0])

        total_volume: float = 0  # Track total volume for summary

        for day in range(start_day, end_day + 1):
            prices_so_far: ndarray = self.price_history[:, : day]
            if config is not None and instruments_to_test is not None:
                new_positions: ndarray = self.getMyPosition(prices_so_far, config, instruments_to_test)
            else:
                new_positions: ndarray = self.getMyPosition(prices_so_far)
            current_prices: ndarray = prices_so_far[:, -1]
            position_limits: ndarray = np.array(
                [int(x) for x in INSTRUMENT_POSITION_LIMIT / current_prices]
            )
            adjusted_positions: ndarray = np.clip(
                new_positions, -position_limits, position_limits
            )
            delta_positions: ndarray = adjusted_positions - current_positions
            volumes: ndarray = current_prices * np.abs(delta_positions)
            daily_volume: float = np.sum(volumes)
            total_volume += daily_volume
            capital_utilisation: float = daily_volume / (
                    INSTRUMENT_POSITION_LIMIT * NUMBER_OF_INSTRUMENTS
            )
            daily_capital_utilisation_list.append(capital_utilisation)
            commission: float = daily_volume * COMMISSION_RATE if self.enable_commission else 0
            cash -= current_prices.dot(delta_positions) + commission
            current_positions = np.array(adjusted_positions)
            for instrument in range(0, 50):
                position_history[instrument].append(current_positions[instrument])
            positions_value: float = current_positions.dot(current_prices)
            profit_and_loss: float = cash + positions_value - portfolio_value
            daily_pnl_list.append(profit_and_loss)
            if day > start_day + 1:
                for instrument in range(0, 50):
                    delta_price: float = (
                            prices_so_far[instrument, -1] - prices_so_far[instrument, -2]
                    )
                    position: int = position_history[instrument][-2]
                    instrument_return: float = delta_price * position
                    instrument_returns[instrument].append(instrument_return)
            for instrument_no in range(0, 50):
                if new_positions[instrument_no] != requested_positions_history[instrument_no][-1]:
                    delta: int = new_positions[instrument_no] - requested_positions_history[
                        instrument_no][-1]
                    new_trade: Trade = {
                        "price_entry": current_prices[instrument_no],
                        "order_type": "buy" if delta > 0 else "sell",
                        "day": day
                    }
                    trades[instrument_no].append(new_trade)
                    requested_positions_history[instrument_no].append(new_positions[instrument_no])
            portfolio_value = cash + positions_value

        backtester_results: BacktesterResults = {
            "daily_pnl": np.array(daily_pnl_list),
            "daily_capital_utilisation": np.array(daily_capital_utilisation_list),
            "daily_instrument_returns": np.array([instrument_returns[i] for i in range(0, 50)]),
            "trades": trades,
            "start_day": start_day,
            "end_day": end_day,
            "total_volume": total_volume  # Added for summary
        }
        return backtester_results

    def save_results(self, results: BacktesterResults, output_file: str = "backtest_results.csv") -> None:
        """
        Save backtest results to a CSV file with summary metrics, daily P&L, and trade history.
        """
        # Summary metrics
        win_rate_pct: float = np.sum(results["daily_pnl"] > 0) / len(results["daily_pnl"]) * 100
        mean_pnl: float = results["daily_pnl"].mean()
        std_dev_pnl: float = results["daily_pnl"].std()
        ann_sharpe: float = np.sqrt(250) * mean_pnl / std_dev_pnl if std_dev_pnl != 0 else 0
        score: float = mean_pnl - 0.1 * std_dev_pnl
        avg_capital_util: float = results["daily_capital_utilisation"].mean() * 100
        summary_data = {
            "start_day": [results["start_day"]],
            "end_day": [results["end_day"]],
            "mean_pnl": [mean_pnl],
            "std_dev_pnl": [std_dev_pnl],
            "ann_sharpe": [ann_sharpe],
            "score": [score],
            "win_rate_pct": [win_rate_pct],
            "total_volume": [results["total_volume"]],
            "avg_capital_utilization_pct": [avg_capital_util]
        }
        summary_df = pd.DataFrame(summary_data)

        # Daily P&L
        days = np.arange(results["start_day"], results["end_day"] + 1)
        daily_pnl_df = pd.DataFrame({
            "day": days,
            "pnl": results["daily_pnl"]
        })

        # Trade history
        trade_data = []
        for instrument_no in range(50):
            for trade in results["trades"][instrument_no]:
                trade_data.append({
                    "instrument": instrument_no,
                    "day": trade["day"],
                    "price_entry": trade["price_entry"],
                    "order_type": trade["order_type"]
                })
        trade_df = pd.DataFrame(trade_data)

        # Save to CSV
        with open(output_file, 'w') as f:
            f.write("Summary Metrics\n")
            summary_df.to_csv(f, index=False)
            f.write("\nDaily P&L\n")
            daily_pnl_df.to_csv(f, index=False)
            f.write("\nTrade History\n")
            trade_df.to_csv(f, index=False)
        print(f"Backtest results saved to {output_file}")

    def show_dashboard(self, backtester_results: BacktesterResults, graphs: List[str]) -> None:
        fig, axs = plt.subplots(2, 2, figsize=(18, 8))
        axs[0][0] = generate_stats_subplot(backtester_results, axs[0][0], self.enable_commission)
        axs[0][1] = get_subplot(graphs[0], backtester_results, axs[0][1])
        if len(graphs) > 1:
            axs[1][0] = get_subplot(graphs[1], backtester_results, axs[1][0])
        else:
            axs[1][0].axis("off")
        if len(graphs) > 2:
            axs[1][1] = get_subplot(graphs[2], backtester_results, axs[1][1])
        else:
            axs[1][1].axis("off")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.suptitle("Backtest Performance Summary", fontsize=16, fontweight="bold")
        plt.show()

    def show_price_entries(self, backtester_results: BacktesterResults) -> None:
        prices_list: List[ndarray] = [
            self.price_history[instrument_no][backtester_results["start_day"] - 1:
                                              backtester_results["end_day"]] for instrument_no in range(0, 50)
        ]
        prices: ndarray = np.array(prices_list)
        days: ndarray = np.arange(backtester_results["start_day"] - 1, backtester_results["end_day"])
        instrument_trades: List[List[Trade]] = [
            backtester_results["trades"][instrument_no] for instrument_no in range(0, 50)
        ]
        buy_entry_prices: List[List[float]] = [[] for i in range(0, 50)]
        buy_entry_days: List[List[int]] = [[] for i in range(0, 50)]
        sell_entry_prices: List[List[float]] = [[] for i in range(0, 50)]
        sell_entry_days: List[List[int]] = [[] for i in range(0, 50)]
        for instrument_no in range(0, 50):
            for trade in instrument_trades[instrument_no]:
                if trade["order_type"] == "buy":
                    buy_entry_prices[instrument_no].append(trade["price_entry"])
                    buy_entry_days[instrument_no].append(trade["day"])
                else:
                    sell_entry_prices[instrument_no].append(trade["price_entry"])
                    sell_entry_days[instrument_no].append(trade["day"])
        fig, ax = plt.subplots(figsize=(14, 6))
        instrument_no = 0
        line, = ax.plot(
            days, prices[instrument_no], color="blue", linestyle="--", linewidth=2,
            label="Instrument Price", zorder=1
        )
        ax.set_xlabel("Days", fontsize=10)
        ax.set_ylabel("Price ($)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        buy_scatter = ax.scatter(
            buy_entry_days[instrument_no], buy_entry_prices[instrument_no],
            color="green", marker="^", s=100, label="Long Entry", zorder=3
        )
        sell_scatter = ax.scatter(
            sell_entry_days[instrument_no], sell_entry_prices[instrument_no],
            color="red", marker="v", s=100, label="Short Entry", zorder=3
        )
        ax.set_title(f"Instrument #{instrument_no} Buys/Sells")

        def on_key(event):
            nonlocal instrument_no
            if event.key == 'right':
                instrument_no = (instrument_no + 1) % len(prices)
            elif event.key == 'left':
                instrument_no = (instrument_no - 1) % len(prices)
            else:
                return
            line.set_ydata(prices[instrument_no])
            buy_scatter.set_offsets(
                np.column_stack((buy_entry_days[instrument_no], buy_entry_prices[instrument_no]))
            )
            sell_scatter.set_offsets(
                np.column_stack((sell_entry_days[instrument_no], sell_entry_prices[instrument_no]))
            )
            ax.set_title(f"Instrument #{instrument_no} Buys/Sells")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.show()

# MAIN EXECUTION #################################################################################
def main() -> None:
    params: Params = parse_command_line_args()
    backtester: Backtester = Backtester(params)
    backtester_results: BacktesterResults = backtester.run(params.start_day, params.end_day)
    backtester.save_results(backtester_results)  # Save results to CSV
    backtester.show_dashboard(backtester_results, params.graphs)
    backtester.show_price_entries(backtester_results)

if __name__ == "__main__":
    main()
