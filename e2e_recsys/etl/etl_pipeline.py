# This file is a proxy for large-scale feature engineering that would be done in Spark or a Data Warehouse # noqa: E501
import pandas as pd

from e2e_recsys.etl.data_split import DataSplitter
from e2e_recsys.etl.gen_negatives import NegativeSampler

ARTICLES_FILEPATH = "/Users/selvino/e2e-recsys/data/articles.csv"
CUSTOMERS_FILEPATH = "/Users/selvino/e2e-recsys/data/customers.csv"
TRANSACTIONS_FILEPATH = "/Users/selvino/e2e-recsys/data/transactions_train.csv"
TRAIN_DATES = ("2018-09-20", "2018-10-20")
VAL_DATES = ("2018-10-21", "2018-11-04")

NEGATIVES_PROPORTION = 1

OUTPUT_TRAIN_PATH = "/Users/selvino/e2e-recsys/data/train_data.csv"
OUTPUT_VAL_PATH = "/Users/selvino/e2e-recsys/data/val_data.csv"

articles = pd.read_csv(ARTICLES_FILEPATH)
customers = pd.read_csv(CUSTOMERS_FILEPATH)
transactions = pd.read_csv(TRANSACTIONS_FILEPATH)


def join_metadata(
    transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    joined_transactions = transactions.merge(articles, on="article_id")
    joined_transactions = joined_transactions.merge(
        customers, on="customer_id"
    )
    return joined_transactions


def save_data(data: pd.DataFrame, output_path: str):
    data.to_csv(output_path, index=False)


# ETL is Split Data -> Gen Negatives -> Join with Articles/Customers
print("Splitting data")
data_splitter = DataSplitter(TRAIN_DATES, VAL_DATES)
train, val = data_splitter.split_data(transactions)
print("Generating Negatives")
train_ns = NegativeSampler(train, NEGATIVES_PROPORTION)
val_ns = NegativeSampler(val, NEGATIVES_PROPORTION)
train_data = train_ns.add_random_data()
val_data = val_ns.add_random_data()
print("Joining to articles/customers")
train_data = join_metadata(train_data, articles, customers)
val_data = join_metadata(val_data, articles, customers)

save_data(train_data, OUTPUT_TRAIN_PATH)
save_data(val_data, OUTPUT_VAL_PATH)
