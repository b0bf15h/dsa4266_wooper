import pandas as pd
import numpy as np
import json
import gzip
import os
import pathlib
import scipy.stats
import re
from pathlib import Path


class DataParsing(object):
    """Reads in zipped json.gz and produces dataframe with each row an individual read"""

    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.data_path = raw_data

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

    def unlabelled_data(self, unzip: bool = True) -> pd.DataFrame:
        """
        Returns dataframe where each row is a read
        """
        # Decompressing .json.gz into a json file
        if unzip:
            output_file = "data.json"
            with gzip.open(self.raw_data, "rb") as gzipped_file:
                with open(output_file, "wb") as json_file:
                    # Read the compressed data and write it to the output file
                    json_data = gzipped_file.read()
                    json_file.write(json_data)

        print("step 1 complete")

        data = []
        if unzip:
            with open("data.json", "r") as file:
                for line in file:
                    data.append(json.loads(line))
            print("step 2 complete")
            os.remove("data.json")
        else:
            with open(self.data_path / "data.json", "r") as file:
                for line in file:
                    data.append(json.loads(line))
            print("step 2 complete")

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
        self.numerical = [
            "dwell_time",
            "sd",
            "mean",
            "m1_dtime",
            "m1_sd",
            "m1_mean",
            "p1_dtime",
            "p1_sd",
            "p1_mean",
        ]
        self.group = [
            "transcript_id",
            "transcript_position",
            "sequence",
            "m1_seq",
            "p1_seq",
        ]

    def find_max_abs_diff(self, row):
        return max(abs(row["m1_mean"] - row["mean"]), abs(row["p1_mean"] - row["mean"]))

    def summarise(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: calculates mean and variance of each float feature
        """
        df_mean = self.df.groupby(self.group)[self.numerical].mean().reset_index()
        df_mean = df_mean.rename(
            columns={
                "dwell_time": "dwell_time_mean",
                "sd": "sd_mean",
                "mean": "mean_mean",
                "m1_dtime": "m1_dtime_mean",
                "m1_sd": "m1_sd_mean",
                "m1_mean": "m1_mean_mean",
                "p1_dtime": "p1_dtime_mean",
                "p1_sd": "p1_sd_mean",
                "p1_mean": "p1_mean_mean",
            }
        )
        df_var = (
            self.df.groupby(self.group)[self.numerical]
            .var()
            .reset_index()[self.numerical]
        )
        df_var = df_var.rename(
            columns={
                "dwell_time": "dwell_time_var",
                "sd": "sd_var",
                "mean": "mean_var",
                "m1_dtime": "m1_dtime_var",
                "m1_sd": "m1_sd_var",
                "m1_mean": "m1_mean_var",
                "p1_dtime": "p1_dtime_var",
                "p1_sd": "p1_sd_var",
                "p1_mean": "p1_mean_var",
            }
        )
        new_df = pd.concat([df_mean, df_var], axis=1)
        count = self.df.groupby(self.group).count().reset_index()["sd"]
        new_df["count"] = count
        new_df["count"] = new_df["count"].astype(float)
        new_df["mean_lower_bound"] = new_df["mean_mean"] - 1.96 * new_df["sd_mean"]
        new_df["mean_upper_bound"] = new_df["mean_mean"] + 1.96 * new_df["sd_mean"]
        print("DATA SUMMARISATION SUCCESSFUL")
        new_df = new_df.fillna(0)
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
        columns_to_drop = ["start", "end"]
        for column in columns_to_drop:
            if column in merged_data.columns:
                merged_data.drop(columns=column, inplace=True)

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
        # weird stuff for ds2
        # data = merged_data['transcript_position'].copy(deep = True)
        # merged_data['relative_sequence_position'] = data
        # merged_data['relative_sequence_position'] = merged_data['relative_sequence_position'].astype(float)
        merged_data["relative_sequence_position"] = np.round(
            (merged_data["transcript_position"].astype(float))
            / merged_data["transcript_length"],
            5,
        )
        print("DATA MERGING SUCCESSFUL")
        return merged_data

    def drop_unused_features(self, merged_data):
        cols_to_drop = [
            "ensembl_gene_id",
            "start_position",
            "end_position",
            "strand",
            "transcription_start_site",
            "transcript_count",
            "percentage_gene_gc_content",
            "gene_biotype",
            "transcript_biotype",
        ]
        for column in cols_to_drop:
            if column in merged_data.columns:
                merged_data.drop(columns=column, inplace=True)
        return merged_data
    def truncate_string(self, input_string):
        match = re.search(r'\.\d+$', input_string)
        if match:
            return input_string[:match.start()]
        return input_string
    def write_data_for_R(
        self,
        data_type: str = "labelled",
        df_name: str = "interm.pkl",
        csv_name: str = "bmart.csv",
    ):
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
        bmart['transcript_id'] = bmart['transcript_id'].astype(str)
        bmart['transcript_id'] = bmart['transcript_id'].apply(self.truncate_string)
        bmart.to_csv(self.data_path / csv_name)
        df.to_pickle(self.data_path / df_name)
