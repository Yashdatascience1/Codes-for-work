import logging
import pandas as pd

#configure logger
logger = logging.getLogger('ReadingFile')
logger.setLevel(logging.INFO)

#Log message formatter
log_format = logging.Formatter("[%(asctime)s]:%(name)s:%(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

#add file handler
file_handler = logging.FileHandler('my_first_log.txt',mode='w')
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)


#Read a file
def read_a_file(file_path):
    df = pd.read_csv(file_path)
    logger.info(f"Read the csv file successfully from {file_path}")


if __name__ == "__main__":
    read_a_file(r"C:\Users\yashs\OneDrive\Desktop\Work codes\Progati\SOQ Script refactoring\Step 4 refactoring\Data.csv")
