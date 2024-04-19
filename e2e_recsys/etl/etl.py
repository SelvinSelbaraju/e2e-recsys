# This file is a proxy for large-scale feature engineering that would be done in Spark or a Data Warehouse
from typing import Tuple
import pandas as pd

ARTICLES_FILEPATH = "/Users/selvino/e2e-recsys/data/articles.csv"
CUSTOMERS_FILEPATH = "/Users/selvino/e2e-recsys/data/customers.csv"
TRANSACTIONS_FILEPATH = "/Users/selvino/e2e-recsys/data/transactions_train.csv"
TRAIN_DATES = ("2018-09-20", "2018-10-20")
VAL_DATES = ("2018-10-21", "2018-11-04")

OUTPUT_TRAIN_PATH = "/Users/selvino/e2e-recsys/data/train_data.csv"
OUTPUT_VAL_PATH = "/Users/selvino/e2e-recsys/data/val_data.csv"

articles = pd.read_csv(ARTICLES_FILEPATH)
customers = pd.read_csv(CUSTOMERS_FILEPATH)
transactions = pd.read_csv(TRANSACTIONS_FILEPATH)

def date_filter(data: pd.DataFrame, dates: Tuple[str, str], date_col: str = "t_dat") -> pd.DataFrame:
    return data[(data[date_col] >= dates[0]) & (data[date_col] <= dates[1])]

def build_data(transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame, dates: Tuple[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transactions = date_filter(transactions, dates)
    joined_transactions = transactions.merge(articles, on="article_id")
    joined_transactions = joined_transactions.merge(customers, on="customer_id")
    return joined_transactions

def save_data(data: pd.DataFrame, output_path: str):
    data.to_csv(output_path, index=False)


train_data = build_data(transactions, articles, customers, TRAIN_DATES)
save_data(train_data, OUTPUT_TRAIN_PATH)

val_data = build_data(transactions, articles, customers, VAL_DATES)
save_data(val_data, OUTPUT_VAL_PATH)

