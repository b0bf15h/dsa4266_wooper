from pathlib import Path
import pandas as pd


class DownSampler(object):
    """Samples data for smaller scale experimentation"""

    def __init__(self, data_path: Path, train_file: str, test_file: str):
        """Initialize the DownSampler

        Args:
            data_path (Path): path which contains data, also where output is written
            train_file (str): filename of training data to sample from
            test_file (str): filename of test data to sample from
        """
        self.data_path = data_path
        self.train = pd.read_pickle(self.data_path / train_file)
        self.test = pd.read_pickle(self.data_path / test_file)
        self.train_sampled = []
        self.test_sampled = []

    def sample_data_n(self, sample_size: list[int]):
        """Sample data using specified list of samples from train,test

        Args:
            sample_size (list[int]): At pos0, specify number of rows to sample from train. At pos1, specify number of rows to sample from test
        """
        train_n = sample_size[0]
        test_n = sample_size[1]
        self.train_sampled.append(self.train.sample(n=train_n, random_state=42))
        self.test_sampled.append(self.test.sample(n=test_n, random_state=42))
        l = len(self.test_sampled) - 1
        self.train = pd.concat(
            [self.train, self.train_sampled[l], self.train_sampled[l]]
        ).drop_duplicates(keep=False)
        self.test = pd.concat(
            [self.test, self.test_sampled[l], self.test_sampled[l]]
        ).drop_duplicates(keep=False)

    def sample_data_p(self, sample_size: list[float]):
        """Sample data using specified list of proportions from train,test

        Args:
            sample_size (list[float]): At pos0, specify proportion to sample from train. At pos1, specify proportion to sample from test
        """
        train_p = sample_size[0]
        test_p = sample_size[1]
        self.train_sampled.append(self.train.sample(frac=train_p, random_state=42))
        self.test_sampled.append(self.test.sample(frac=test_p, random_state=42))
        l = len(self.test_sampled) - 1
        self.train = pd.concat(
            [self.train, self.train_sampled[l], self.train_sampled[l]]
        ).drop_duplicates(keep=False)
        self.test = pd.concat(
            [self.test, self.test_sampled[l], self.test_sampled[l]]
        ).drop_duplicates(keep=False)

    def sample_data(self, mode: str, sample_size: list, num_samples: int):
        """
        Samples data independently from train and test data, using number of samples or proportion'
        mode = n for specifying number of rows and mode = p for specifying proportion
        """
        if type(mode) != str:
            print("Mode must be a string")
        if mode == "n":
            for i in range(num_samples):
                self.sample_data_n(sample_size)
                print(f"Training data has {len(self.train)} rows")
                print(f"Testing data has {len(self.test)} rows")
            return
        for i in range(num_samples):
            self.sample_data_p(sample_size)
            print(f"Training data has {len(self.train)} rows")
            print(f"Testing data has {len(self.test)} rows")

    def write_output(self):
        for i in range(len(self.train_sampled)):
            train_name = "downsample_train_" + str(i + 1) + ".pkl"
            test_name = "downsample_test" + str(i + 1) + ".pkl"
            self.train_sampled[i].to_pickle(self.data_path / train_name)
            self.test_sampled[i].to_pickle(self.data_path / test_name)


# if __name__ == "__main__":
#     Downsampler = DownSampler(Path(__file__).resolve().parents[1] / "data", 'train.pkl', 'test.pkl' )
#     Downsampler.sample_data("n", [8000, 2000], 3)
#     Downsampler.write_output()
