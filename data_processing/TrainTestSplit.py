import pandas as pd
import pathlib


class TrainTestSplit(object):
    """
    Performs train-test split of data
    """

    def __init__(self, merged_pkl_data: pd.DataFrame):
        self.merged_pkl_data = merged_pkl_data

    def train_test_split(
        self,
        train_size: float,
        data_path: pathlib.Path,
        train_output_name: str,
        test_output_name: str,
    ):
        """
        Group data by gene id when splitting to avoid data leakage
        """
        grouped_gene_id = pd.Series(
            self.merged_pkl_data.groupby(by="gene_id")["transcript_id"]
            .count()
            .sort_values()
        )
        Grouped_id = pd.qcut(grouped_gene_id, q=10, labels=False)
        Grouped_id = pd.DataFrame(
            {"Gene_ID": grouped_gene_id.index, "Quantile": Grouped_id}
        )

        train_gene_id = []
        for i in range(0, 10):
            group = Grouped_id[Grouped_id["Quantile"] == i]
            selected = group["Gene_ID"].sample(
                n=int(train_size * len(group)), random_state=42
            )
            for item in selected:
                train_gene_id.append(item)

        train_data = self.merged_pkl_data[
            self.merged_pkl_data["gene_id"].isin(train_gene_id) == True
        ]
        test_data = self.merged_pkl_data[
            self.merged_pkl_data["gene_id"].isin(train_gene_id) != True
        ]
        print("TRAIN-TEST SPLIT SUCCESSFUL")
        train_data.to_pickle(data_path / train_output_name)
        test_data.to_pickle(data_path / test_output_name)
        return (train_data, test_data)
