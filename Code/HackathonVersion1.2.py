from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import re
from threading import Thread


class DataMiner(object):
    def __init__(self, html_data, test_data, train_data):
        self.common_keys = ['', 'https:', 'http:', 'id', 'php', 'www', 'development', 'clinical', 'australia', 'page',
                            'view', 'com', 'd', 'and', 'html', 'index', 'about', 'technology', 'new', 'e', 'we', 'in',
                            'research', 'org', 'do', 'en', 'results', 'reports', 'content', 'injury', 'international',
                            'radiation', 'surgical', 'anti', 'i', 'colorectal', 's', 'are', 'de', 'is', 'no', 'htm',
                            'bone', 'diseases', 'nih', 'heart', 'co', 'studies', 'node', 'details', 'imaging', 'asp',
                            'human', 'gut', 'what', 'head', 'diabetes', 'science', 'a', 'z', 'best', 'au', 'access',
                            'american', 'medicine', 'edu', 'of', 'cancer', 'medical', 'managing', 'search', 'patient',
                            'item', 'gene', 'advanced', 'nutrition', 'disorders', 'hospital', 'b', 'gov', 'on', 'us',
                            'health', 'with', 'healthcare', 'safety', 'source', 'case', 'blood', 'surgery', 'care',
                            'family', 'information', 'at', 'effects', 'national', 'c', 'uk', 'diagnostic', 'true',
                            'young', 'primary', 'doctors', 'neck', 'how', 'an', 'treatment', 'breast', 'emergency',
                            'main', 'full', 'policy', 'site', 'p', 'line', 'report', 'your', 'ca', 'disease', 'for',
                            'associated', 'section', 'long', 'general', 'pain', 'aspx', 'net', 'by', 'liver', 'f',
                            'the', 'resources', 'lung', 'oncology', 'to', 'response', 'perspective', 'drug',
                            'secondary', 'society', 'online', 'nuclear', 'current', 'tool', 'molecular', 'dr',
                            'children', 'recent', 'death', 'adult', 'open', 'chemotherapy', 'onset', 'biomedcentral',
                            'food', 'therapies', 'years', 'may', 'historical', 'diagnosis', 'arthritis', 'risk',
                            'small', 'articles', 'among', 'childhood', 'service', 'support', 'change', 'ovarian',
                            'media', 'control', 'block', 'birth', 'training', 'pregnancy', 'ch', 'syndrome', 'active',
                            'campaign', 'women', 'foundation', 'data', 'impact', 'oral', 'from', 'transplant', 'level',
                            'call', 'l', 'management', 'qa', 'archive', 'survival', 'you', 'term', 'vein', 'metastatic',
                            'therapy', 'approach', 'time', 'leukemia', 'guidance', 'peripheral', 'community', 'rate',
                            'its', 'age', 'action', 'life', 'our', 'guide', 'first', 'better', 'financial',
                            'effectiveness', 'foot', 'statements', 'system', 'related', 'high', 'non', 'endocrine',
                            'physicians', 'childrens', 'review', 'be', 'A', 'people', 'body', 'sleep', 'following', 't',
                            'colon', 'into', 'factor', 'up', 'providers', 'planning', 'their', 'men', 'targeted',
                            'congress', 'implications', 'reproductive', 'quality', 'stress', 'targets', 'as',
                            'conference', 'analysis', 'm', 'failure', 'association', 'early', 'follow', 'anxiety',
                            'depression', 'medium', 'pregnant', 'low', 'med', 'study', 'using', 'therapeutic',
                            'antibodies', 'issue', 'during', 'improvement', 'part', 'endo', 'screening', 'methods',
                            'biology', 'files', 'article', 'protein', 'services', 'activity', 'update', 'based',
                            'progress', 'partners', 'multiple', 'cancers', 'ways', 'q', 'evidence', 'list', 'type',
                            'detection', 'use', 'y', 'lessons', 'end', 'j', 'pediatric', 'all', 'mit', 'role', 'major',
                            'april', 'nhs', 'video', 'thyroid', 'or', 'problems', 'critical', 'hiv', 'act', 'internal',
                            'who', 'global', 'work', 'australian', 'effective', 'should', 'bio', 'transfer', 'symptoms',
                            'vascular', 'day', 'loss', 'board', 'staff', 'genetics', 'kidney', 'college', 'central',
                            'not', 'so', 'stem', 'over', 'profiles', 'show', 'south', 'scientific', 'cell', 'stage',
                            'self', 'sun', 'after', 'journal', 'x', 'practices', 'spine', 'field', 'cells',
                            'combination', 'brain', 'infection', 'et', 'editorial', 'outcomes', 'water', 'real', 'abs',
                            'phase', 'cost', 'ac', 'healthy', 'world', 'patients', 'survey', 'reduce', 'large']
        self.train_data = pd.read_csv(train_data)
        self.test_data = pd.read_csv(test_data)
        self.categories = None
        self.categories_filter = {}
        self.train_data = self.train_model(self.train_data, "train")
        self.test_data = self.train_model(self.test_data)
        self.train_data = self.covert_string_data_to_number(self.train_data)
        d_c_tree = DecisionTreeRegressor()
        d_c_tree.fit(self.train_data[["Url"]], self.train_data[["Tag"]])
        predicted_Values = d_c_tree.predict(self.test_data[["Url"]])
        self.test_data.append(pd.DataFrame(predicted_Values).round())
        self.test_data = self.convert_number_to_string(self.test_data)
        self.test_data = self.test_data.sort_values(['Url'], ascending=[False])
        #del self.test_data[["Domain"]]
        del self.test_data[["Unamed 0"]]
        self.test_data = self.test_data.rename(columns={"Url": "Tag"})
        self.test_data.to_csv("result_new9.csv")

    def convert_number_to_string(self, data):
        for index, value in data.iterrows():
            for sub_index, sub_value in enumerate(self.categories):
                if data.at[index, "Url"] == sub_index:
                    data.at[index, "Url"] = sub_value
        return data

    def covert_string_data_to_number(self, data):
        for index, value in data.iterrows():
            for sub_index, sub_value in enumerate(self.categories):
                if data.at[index, "Tag"] == sub_value:
                    data.at[index, "Tag"] = sub_index
        return data

    def train_model(self, data, tag=None):
        if tag:
            print("[INFO]: Collecting Data For Training Started")
            self.categories = self.get_categories(data)
            for categorise in self.categories:
                self.categories_filter[categorise] = self.data_distributor(data, categorise)
            print("[INFO]: Collecting Data For Training Done")
        print("[INFO]: Modifying Data Started")
        for index, row in data.iterrows():
            count = 0
            for key, value in self.categories_filter.items():
                if key in row.Url or count < len(set(re.split('/|_|-|&|\?|%|$|#|@|\.|=', row.Url)) & set(value)):
                    count = len(set(re.split('/|_|-|&|\?|%|$|#|@|\.|=', row.Url)) & set(value))
                    data.at[index, "Url"] = key
            if count == 0:
                data.at[index, "Url"] = np.NaN
            for sub_index, key in enumerate(self.categories):
                if data.at[index, "Url"] == key:
                    data.at[index, "Url"] = sub_index
        for index, row in data.iterrows():
            if data.at[index, "Url"] not in range(0, len(self.categories)):
                data.at[index, "Url"] = data.Url.median()
        print("[INFO]: Modifying Data Done")
        return data

    def data_distributor(self, data, tag):
        my_list = []
        useful_data = []
        waste_data = []
        for index, row in data.iterrows():
            if row.Tag == tag:
                my_list = my_list + re.split('/|_|-|&|\?|%|$|#|@|\.|=', row.Url)
        common_list = Counter(my_list).most_common()
        for value in common_list:
            if len(value[0]) > 3 and value[0] not in self.common_keys:
                if re.findall('\d+', value[0]):
                    waste_data.append(value[0])
                else:
                    useful_data.append(value[0])
        return useful_data[0:9]

    @staticmethod
    def get_categories(data):
        categories = []
        for index, row in data.iterrows():
            categories.append(row.Tag)
        return list(set(categories))


if __name__ == '__main__':
    ROOT = DataMiner("html_data_reduced.csv", "test_nvPHrOx.csv", "train.csv")
