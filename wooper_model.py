from data_processing.DataProcessing import DataParsing, MergeData


class WooperModel(object):
    def __init__(self):
        self.raw_data = []
        self.raw_info = []

    # Task 1
    def train_model(self, raw_data, raw_info):
        self.raw_data = raw_data
        self.raw_info = raw_info
        parsed_data = DataParsing(self.raw_data).unlabelled_data()
        merged_data = MergeData(parsed_data, self.raw_info).merge()
        print("length of merged data: ")
        print(len(merged_data))


if __name__ == "__main__":
    model_instance = WooperModel()

    model_instance.train_model(
        "/Users/charlenechan/dsa4266_wooper/dataset/dataset0.json.gz",
        "/Users/charlenechan/dsa4266_wooper/dataset/data.info",
    )
