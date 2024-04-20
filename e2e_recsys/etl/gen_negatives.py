from typing import List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# What proportion of the dataset to add in as synthetic negatives
NEGATIVES_PROPORTION = 2

class NegativeSampler:
    def __init__(self, data: pd.DataFrame, negatives_proportion: float, date_col: str = "t_dat", customer_id_col: str = "customer_id", article_id_col: str = "article_id", target_col: str = "purchased"):
        self.data = data
        self.target_col = target_col
        self.num_negatives = round(len(self.data) * negatives_proportion)
        self.date_col = date_col

        # Set candidates for sampling date, customer id and article id
        self._get_candidate_dates()
        self.candidate_customers = self.data[customer_id_col].unique()
        self.candidate_articles = self.data[article_id_col].unique()

        # Set the target to 1 for all of the original data
        self.data[self.target_col] = 1    

    def _get_candidate_dates(self) -> List[str]:
        start_date = datetime.strptime(self.data[self.date_col].min(), "%Y-%m-%d")
        end_date = datetime.strptime(self.data[self.date_col].max(), "%Y-%m-%d")

        self.candidate_dates = np.array([])
        # Iterate through the range of dates and append them to the list
        current_date = start_date
        while current_date <= end_date:
            self.candidate_dates = np.append(self.candidate_dates, current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

    def add_random_data(self):
        negatigves_df_dict = {
            "t_dat": np.random.choice(self.candidate_dates, size=self.num_negatives, replace=True),
            "customer_id": np.random.choice(self.candidate_customers, size=self.num_negatives, replace=True),
            "article_id": np.random.choice(self.candidate_articles, size=self.num_negatives, replace=True),
            "sales_channel_id": np.ones(shape=(self.num_negatives,)),
            # Set the negatives to have a target of 0
            self.target_col: np.zeros(shape=(self.num_negatives,)),
        }
        # This ensures the indices are unique between the two dataframes
        negatives = pd.DataFrame(negatigves_df_dict, index = range(len(self.data), len(self.data) + self.num_negatives))
        return pd.concat([self.data, negatives])

