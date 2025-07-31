import logging
import os

log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "backtest.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w")]
)

logger = logging.getLogger("backtest_logger")

def clear_backtest_log():
    """Efficiently clears the contents of the backtest log file."""
    with open(log_path, 'w'):
        pass

if __name__ == "__main__":
    clear_backtest_log()
