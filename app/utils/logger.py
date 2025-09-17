import logging
import sys

# 格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# 設定
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def get_logger(name: str):
    logger = logging.getLogger(name)

    # 避免重複 handler
    if not logger.handlers:
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # File handler (可選)
        # file_handler = logging.FileHandler("app.log")
        # file_handler.setLevel(logging.WARNING) # or ERROR
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger
