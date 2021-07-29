import argparse
import random
from pathlib import Path

from dataset import Dataset
from utils import seed


def languageDetection(filePath: Path):
    return 'pred'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Languages Detection.')
    parser.add_argument('--action', default='train', choices=['train', 'eval'], type=str,
                        help='an integer for the accumulator')
    parser.add_argument('--data_path', default=r"Languages/data/", type=str, help='Path to the data')
    parser.add_argument('--data_statistics', default=True, type=bool, help='Print Statistics on the dataset')
    parser.add_argument('--seed', default=random.randint(0, 2 ** 10), type=int, help='Seed for this run')
    parser.add_argument('--file_to_predict', type=str, help='Path to the file we want to predict')
    args = parser.parse_args()

    if args.action == 'train':
        seed(args.seed)

        dataset = Dataset(path=args.data_path, stats=args.data_statistics)
        dataset.load_data()
        dataset.split_data()

        # TODO implement the classifier

    elif args.action == 'eval':
        file_to_predict = Path(args.file_to_predict)
        languageDetection(file_to_predict)