import pandas as pd

class PreProcess:
    def __init__(self):
        self.data = None

    def read_data(self, path):
        self.data = pd.read_csv(path)

    def drop_columns(self, columns):
        self.data = self.data.drop(columns, axis=1)

    def filter_rows(self, column, value):
        self.data = self.data[self.data[column] == value]

    def fill_na(self, column, value):
        self.data[column].fillna(value, inplace=True)

    def sum(self,column,columns):
        self.data[column] = self.data[columns].sum(axis=1).astype(float)