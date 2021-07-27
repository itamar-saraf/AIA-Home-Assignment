import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    """
    This Class holds all the dataset.
    Including train, validation and test.
    """

    def __init__(self, path: str, stats: bool = True):
        self.data_path = Path(path)
        self.data = pd.DataFrame(columns=['tweet', 'language'])
        self.train = None
        self.val = None
        self.test = None
        self.stats = stats

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

    def split_data(self):
        """
        This function splitting the dataset using sk-learn to 75% train, 25% test.
        from the train, it's taking another 15% to validation set
        """
        self.train, self.test = train_test_split(self.data, test_size=0.25, shuffle=True)
        self.train, self.val = train_test_split(self.train, test_size=0.15, shuffle=True)

    @staticmethod
    def _remove_numbers(tweet: str) -> str:
        """
        This function cleaning tweets from numbers.

        :param tweet: The tweet that need to be cleaned
        :return: Tweet without the digits
        """
        return re.sub(r'[0-9]+', '', tweet)

    @staticmethod
    def _extract_label(language: Path) -> str:
        """
        This functions Extract the label from the path to language folder
        :param language: folder that contains all the tweets of specified language
        :return: label
        """
        return language.name.split('_')[-1]
