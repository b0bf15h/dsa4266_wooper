import pandas as pd
import numpy as np
import json
import gzip
import os
import pathlib
from scipy import stats

class DataParsing(object):
    """Reads in zipped json.gz and produces dataframe with each row an individual read"""

    def __init__(self, raw_data):
        self.raw_data = raw_data

    def find_keys(self, d: dict) -> list:
        """
        Extracts all keys from a dictionary recursively
        """
        keys = []
        for key, value in d.items():
            # recursively adds keys to list
            if isinstance(value, dict):
                keys.append(key)
                keys.extend(self.find_keys(value))
            else:
                keys.append(key)
        return keys

    def parse_seq_pos0_only(self, d: dict):
        """For each read of 9 data points, return lists of their info"""
        keys = self.find_keys(d)
        # extract info from each set of keys including adjacent positions
        transcript_id = keys[0]
        position = keys[1]
        sequences = [keys[2][0:5], keys[2][1:6], keys[2][2:7]]
        reads = d[keys[0]][keys[1]][keys[2]]
        transcript_id = [transcript_id] * len(reads)
        seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean = [], [], [], [], []
        m1_seq, m1_dtime, m1_sd, m1_mean = [], [], [], []
        p1_seq, p1_dtime, p1_sd, p1_mean = [], [], [], []
        for read in reads:
            seq_pos.append(position)
            seq_seq.append(sequences[1])
            seq_dtime.append(read[3])
            seq_sd.append(read[4])
            seq_mean.append(read[5])
            m1_seq.append(sequences[0])
            m1_dtime.append(read[0])
            m1_sd.append(read[1])
            m1_mean.append(read[2])
            p1_seq.append(sequences[2])
            p1_dtime.append(read[6])
            p1_sd.append(read[7])
            p1_mean.append(read[8])
        return (
            transcript_id,
            seq_pos,
            seq_seq,
            seq_dtime,
            seq_sd,
            seq_mean,
            m1_seq,
            m1_dtime,
            m1_sd,
            m1_mean,
            p1_seq,
            p1_dtime,
            p1_sd,
            p1_mean,
        )

    def parse_data_pos0(self, data):
        """Iterate through data and extract data into lists for df creation"""
        (
            transcript_id,
            seq_pos,
            seq_seq,
            seq_dtime,
            seq_sd,
            seq_mean,
            m1_seq,
            m1_dtime,
            m1_sd,
            m1_mean,
            p1_seq,
            p1_dtime,
            p1_sd,
            p1_mean,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
        for entry in data:
            (
                t_id,
                s_pos,
                s_seq,
                s_dtime,
                s_sd,
                s_mean,
                m_seq,
                m_dtime,
                m_sd,
                m_mean,
                p_seq,
                p_dtime,
                p_sd,
                p_mean,
            ) = self.parse_seq_pos0_only(entry)
            transcript_id.extend(t_id)
            seq_pos.extend(s_pos)
            seq_seq.extend(s_seq)
            seq_dtime.extend(s_dtime)
            seq_sd.extend(s_sd)
            seq_mean.extend(s_mean)
            m1_seq.extend(m_seq)
            m1_dtime.extend(m_dtime)
            m1_sd.extend(m_sd)
            m1_mean.extend(m_mean)
            p1_seq.extend(p_seq)
            p1_dtime.extend(p_dtime)
            p1_sd.extend(p_sd)
            p1_mean.extend(p_mean)
        return (
            transcript_id,
            seq_pos,
            seq_seq,
            seq_dtime,
            seq_sd,
            seq_mean,
            m1_seq,
            m1_dtime,
            m1_sd,
            m1_mean,
            p1_seq,
            p1_dtime,
            p1_sd,
            p1_mean,
        )

    def replace_T_with_U(self, seq: str):
        """
        Replaces T bases with U since they are equivalent
        """
        return seq.replace("T", "U")

    def remove_overlapping_m1(self, seq: str):
        return seq[0]

    def remove_overlapping_p1(self, seq: str):
        return seq[4]

    def unlabelled_data(self) -> pd.DataFrame:
        """
        Returns dataframe where each row is a read
        """
        # Decompressing .json.gz into a json file
        output_file = "data.json"
        with gzip.open(self.raw_data, "rb") as gzipped_file:
            with open(output_file, "wb") as json_file:
                # Read the compressed data and write it to the output file
                json_data = gzipped_file.read()
                json_file.write(json_data)
        print("step 1 complete")

        data = []
        with open("data.json", "r") as file:
            for line in file:
                data.append(json.loads(line))
        print("step 2 complete")
        os.remove("data.json")

        (
            transcript_id,
            seq_pos,
            seq_seq,
            seq_dtime,
            seq_sd,
            seq_mean,
            m1_seq,
            m1_dtime,
            m1_sd,
            m1_mean,
            p1_seq,
            p1_dtime,
            p1_sd,
            p1_mean,
        ) = self.parse_data_pos0(data)
        print("step 3 complete")

        df = pd.DataFrame(
            {
                "transcript_id": transcript_id,
                "transcript_position": seq_pos,
                "sequence": seq_seq,
                "dwell_time": seq_dtime,
                "sd": seq_sd,
                "mean": seq_mean,
                "m1_seq": m1_seq,
                "m1_dtime": m1_dtime,
                "m1_sd": m1_sd,
                "m1_mean": m1_mean,
                "p1_seq": p1_seq,
                "p1_dtime": p1_dtime,
                "p1_sd": p1_sd,
                "p1_mean": p1_mean,
            }
        )

        df["sequence"] = df["sequence"].apply(self.replace_T_with_U)
        df["m1_seq"] = df["m1_seq"].apply(self.replace_T_with_U)
        df["p1_seq"] = df["p1_seq"].apply(self.replace_T_with_U)
        # df["m1_seq"] = df["m1_seq"].apply(self.remove_overlapping_m1)
        # df["p1_seq"] = df["p1_seq"].apply(self.remove_overlapping_p1)

        print("DATA PARSING SUCCESSFUL")
        return df


class SummariseDataByTranscript(object):
    """
    Summarise data by calculating summary statistics, grouping by sequence
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numerical_mean = ['dwell_time', 'mean', 'm1_dtime', 'm1_mean', 'p1_dtime','p1_mean']
        self.numerical_var = ['sd', 'm1_sd','p1_sd']
        self.dtime = ["dwell_time", "m1_dtime", "p1_dtime"]
        self.group = ['transcript_id', 'transcript_position', 'sequence', 'm1_seq', 'p1_seq']
        
    def calculate_ci_lower(data):
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
        n = len(data)
        z = stats.t.ppf(0.975, df=n-1)  # For a 95% confidence interval (alpha = 0.05)

        margin_of_error = z * (std_dev / np.sqrt(n))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        return lower_bound
    
    def calculate_ci_upper(self, data):
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
        n = len(data)
        z = stats.t.ppf(0.975, df=n-1)  # For a 95% confidence interval (alpha = 0.05)

        margin_of_error = z * (std_dev / np.sqrt(n))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        return upper_bound

    def summarise(self):
        df_mean = self.df.groupby(self.group)[self.numerical_mean].mean().reset_index()
        df_mean = df_mean.rename(columns={"dwell_time": "dwell_time_mean", "mean":"mean_mean", "m1_dtime": 'm1_dtime_mean', 'm1_mean':"m1_mean_mean", 'p1_dtime':"p1_dtime_mean", 'p1_mean':"p1_mean_mean"})
        df_var = self.df.groupby(self.group)[self.numerical_var].var().reset_index()[self.numerical]
        df_var = df_var.rename(columns={"sd": "sd_sample_var", 'm1_sd': "m1_sd_sample_var", 'p1_sd':"p1_sd_sample_var"})
        df_ci_lower = self.df.groupby(self.group)[self.dtime].agg(self.calculate_ci_lower).reset_index()[self.dtime]
        df_ci_lower = df_ci_lower.rename(columns={"dwell_time": "dwell_time_ci_lower", 'm1_dtime': "m1_dtime_ci_lower", 'p1_dtime':"p1_dtime_ci_lower"})
        df_ci_upper = self.df.groupby(self.group)[self.dtime].agg(self.calculate_ci_upper).reset_index()[self.dtime]
        df_ci_upper = df_ci_upper.rename(columns={"dwell_time": "dwell_time_ci_upper", 'm1_dtime': "m1_dtime_ci_upper", 'p1_dtime':"p1_dtime_ci_upper"})
        
        new_df = pd.concat([df_mean, df_var, df_ci_lower, df_ci_upper], axis=1)
        count = self.df.groupby(self.group).count().reset_index()['sd']
        new_df['count'] = count
        print("DATA SUMMARISATION SUCCESSFUL")
        return new_df


class MergeData(object):
    """
    Merges unlabelled data with either the labels (for model training)
    or with additional features queried from outside sources
    """

    def __init__(self, parsed_data: pd.DataFrame, raw_info, data_path: pathlib.Path):
        self.parsed_data = parsed_data
        self.raw_info = raw_info
        self.data_path = data_path

    def merge_with_labels(self):
        data_info = pd.read_csv(self.raw_info, delimiter="\,")
        data_info["transcript_position"] = data_info["transcript_position"].astype(
            "str"
        )
        merged_data = pd.merge(
            self.parsed_data,
            data_info,
            on=["transcript_id", "transcript_position"],
            how="left",
        )

        print("DATA MERGING SUCCESSFUL")
        return merged_data

    def merge_with_features(self):
        """
        Merge with csv data(queried from R script) on transcript id
        """
        data_info = pd.read_csv(self.raw_info, header=[0])
        if data_info.shape[0] == 0:
            return self.parsed_data
        data_info.rename(
            columns={"ensembl_transcript_id": "transcript_id"}, inplace=True
        )
        merged_data = pd.merge(
            self.parsed_data, data_info, on=["transcript_id"], how="left"
        )
        merged_data["relative_sequence_position"] = np.round(
            (merged_data["transcript_position"].astype(float))
            / merged_data["transcript_length"],
            5,
        )
        outliers = merged_data[merged_data["relative_sequence_position"] >= 1]
        outliers.to_pickle(self.data_path / "outliers_length.pkl")
        print("DATA MERGING SUCCESSFUL")
        return merged_data

    def write_data_for_R(self, data_type: str = "labelled"):
        """
        Writes data to be used for R querying into data path, as well as store intermediate Df(with labels) for later use
        """
        if data_type == "labelled":
            df = self.merge_with_labels()
        elif data_type == "unlabelled":
            df = self.parsed_data
        else:
            print(
                f"{data_type} is not a valid argument, it has to be either labelled or unlabelled"
            )
            return
        bmart = df[["transcript_id", "transcript_position"]]
        bmart.to_csv(self.data_path / "bmart.csv")
        df.to_pickle(self.data_path / "interm.pkl")
