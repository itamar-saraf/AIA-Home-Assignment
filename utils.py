import numpy as np


def seed(random_number: int):
    """
    This function determine the ssed for this run.
    :param random_number: Random number that been generated each run
    """
    np.random.seed(random_number)
    print(f'Seed for this run is {random_number}')
