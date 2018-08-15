import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from html2text import html2text,HTML2Text
import requests
import re


class UrlCategoriseMiner(object):
    def __init__(self, train_file, test_file, extra_file, filtered_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.extra_data = pd.read_csv(extra_file, chunksize=500)
        self.categories = self.get_categories(self.train_data)
        self.extra_filtered_data = self.filtering_html_data(self.test_data,request_required=True)
        #self.extra_filtered_data = pd.read_csv(filtered_file)
        self.train_data = self.train_data.merge(self.extra_filtered_data)
        del self.train_data['Unnamed: 0']
        self.get_common_keywords(self.train_data)

    def get_common_keywords(self,data):
        common_list = []
        for index, row in data.iterrows():
            for index,categories in enumerate(self.categories):
                if categories in row.Html:
                    data.at[index,"Html"] = index
                else:
                    common_list = common_list + re.split('\', \'|\',\"', row.Html)
        print(common_list)

    @staticmethod
    def filtering_html_data(html_data,request_required = False):
        result_data = pd.DataFrame()
        h = HTML2Text()
        h.ignore_links = True
        if not request_required:
            for chunk_data in html_data:
                print("=======START========")
                for index,row in chunk_data.iterrows():
                    filtered_value = set(re.split(' |\n|\*', h.handle(row.Html)))
                    chunk_data.at[index,"Html"] = filtered_value
                print("========END=========")
                result_data = result_data.append(chunk_data)
            return result_data
        else:
            for index,row in html_data.iterrows():
                try:
                    filtered_value = set(re.split(' |\n|\*', h.handle(requests.get(row.Url).text)))
                except:
                    filtered_value = []
                html_data.at[index,"Url"] = filtered_value
            return html_data



    @staticmethod
    def get_categories(data):
        categories = []
        for index, row in data.iterrows():
            categories.append(row.Tag)
        return list(set(categories))


if __name__ == '__main__':
    ROOT = UrlCategoriseMiner("train.csv", "test_nvPHrOx.csv", "html_data(1).csv", "html_data.csv")
