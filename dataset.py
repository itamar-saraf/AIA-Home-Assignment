import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


class Dataset:
    """
    This Class holds all the dataset.
    Including train, validation and test.
    """

    def __init__(self, path: str, stats: bool = True, ngram: int = 3, max_features: int = 500):
        self.data_path = Path(path)
        self.data = pd.DataFrame(columns=['tweet', 'language'])
        self.train = None
        self.test = None
        self.stats = stats
        self.le = LabelEncoder()
        self.cv = CountVectorizer(strip_accents='ascii', analyzer='char', ngram_range=(ngram, ngram),
                                  max_features=max_features)

    def load_data(self):
        """
        This function iterate over all folders in the data path.
        each folder represent a language.
        each language contains tweet that builds our dataset
        """
        languages = [folder for folder in self.data_path.iterdir() if folder.is_dir()]
        for language in languages:
            label = self._extract_label(language)
            for tweet in language.iterdir():
                tweet = self._remove_numbers(tweet.read_text())
                self.data = self.data.append({'tweet': tweet, 'language': label}, ignore_index=True)
        if self.stats:
            print('You can see if the dataset is balanced')
            print(self.data.language.value_counts())
        self._label_encoding()

    def split_data(self):
        """
        This function splitting the dataset using sk-learn to 70% train, 30% test.
        """
        self.train, self.test = train_test_split(self.data, test_size=0.3, shuffle=True)

    def preprocess(self):
        """
        Transforms the data to ngrams,
        also build vocabulary.
        """
        self._char_ngrams()

    def _label_encoding(self):
        """
        creates code labels from languages labeling
        """
        y = self.data["language"]
        self.data['label_code'] = self.le.fit_transform(y)

    def _char_ngrams(self):
        """
        Transforms all tweets to ngram, saving it in the dataset
        """
        x = self.train['tweet']
        x = self.cv.fit_transform(x).toarray()
        self.train.insert(3, 'ngrams', list(x), True)

    @staticmethod
    def _extract_label(language: Path) -> str:
        """
        This functions Extract the label from the path to language folder
        :param language: folder that contains all the tweets of specified language
        :return: label
        """
        return language.name.split('_')[-1]

    @staticmethod
    def _remove_numbers(tweet: str) -> str:
        """
        This function cleaning tweets from numbers.

        :param tweet: The tweet that need to be cleaned
        :return: Tweet without the digits
        """
        return re.sub(r'[0-9]+', '', tweet)
