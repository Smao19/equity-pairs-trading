import logging
import os

# Set log paths for easy access
backtest_log_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "backtest.log")
os.makedirs(os.path.dirname(backtest_log_path), exist_ok=True)
data_handler_log_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "data_handler.log")
os.makedirs(os.path.dirname(data_handler_log_path), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(backtest_log_path, mode="w")]
)

logger = logging.getLogger("backtest_logger")

def clear_log(log_path):
    """Efficiently clears the contents of the backtest log file."""
    with open(log_path, 'w'):
        pass


if __name__ == "__main__":
    clear_log(backtest_log_path)
    clear_log(data_handler_log_path)
