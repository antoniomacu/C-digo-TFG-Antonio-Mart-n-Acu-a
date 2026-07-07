# Preprocessing for cond_reg_v2
#
# Combines:
# - File-level split strategy from bin/ (prevents temporal leakage across day-files)
# - Conditional regressor windowing from cond_reg/ (past window -> current timestep output)

import json
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(
        self,
        parameters,
        seed=42,
        train_split=0.8,
        val_normal_split=0.1,
        abnormal_val_split=0.5,
    ):
        """
        Args:
            parameters: Hyperparameters configuration
            seed: Random seed for reproducibility
            train_split: Fraction of normal data for training (0.8 = 80%)
            val_normal_split: Fraction of normal data for validation (0.1 = 10%)
            abnormal_val_split: Fraction of abnormal data for validation (rest goes to test)
        """
        self.hparams = parameters
        self.seed = seed
        self.train_split = train_split
        self.val_normal_split = val_normal_split
        self.test_normal_split = 1 - train_split - val_normal_split
        self.abnormal_val_split = abnormal_val_split
        self.norm_params_cached: dict | None = None

    def normalize_data(self, data, params, method):
        """Normalize data using z-score or min-max parameters."""
        normalized_data = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col not in params:
                raise Exception(
                    f"Normalization parameters for column '{col}' are not available."
                )

            if method == "zscore":
                mean = params[col]["mean"]
                std = params[col]["std"]
                normalized_data[col] = (data[col] - mean) / std

            elif method == "min-max":
                min_ = params[col]["min"]
                max_ = params[col]["max"]
                normalized_data[col] = (data[col] - min_) / (max_ - min_)

            else:
                raise Exception("Provided normalization method is not valid.")

        return normalized_data

    def denormalize_data(self, data_normalized, params, method):
        """Denormalize data using z-score or min-max parameters."""
        denormalized_data = pd.DataFrame(index=data_normalized.index)

        for col in data_normalized.columns:
            if col not in params:
                raise Exception(
                    f"Denormalization parameters for column '{col}' are not available."
                )

            if method == "zscore":
                mean = params[col]["mean"]
                std = params[col]["std"]
                denormalized_data[col] = data_normalized[col] * std + mean

            elif method == "min-max":
                min_ = params[col]["min"]
                max_ = params[col]["max"]
                denormalized_data[col] = data_normalized[col] * (max_ - min_) + min_

            else:
                raise Exception("Provided denormalization method is not valid.")

        return denormalized_data

    def get_normalization_params(self, data, save_path="norm_params.json"):
        """Calculate and save per-feature normalization parameters."""
        norm_params = {}

        for col in data.columns:
            mean = float(np.mean(data[col]))
            std = float(np.std(data[col]))
            min_ = float(np.min(data[col]))
            max_ = float(np.max(data[col]))

            if col.startswith("pump_id"):
                max_ = 1.0

            norm_params[col] = {
                "mean": mean,
                "std": std,
                "min": min_,
                "max": max_,
            }

        self.norm_params_cached = norm_params

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as file:
            json.dump(norm_params, file, indent=4)

        return norm_params

    def _get_file_list(self, data_path, is_training=False):
        """Get sorted CSV file list from directory."""
        all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])

        if is_training:
            from collections import Counter

            pump_counts = Counter(f.split("_2")[0] for f in all_files)
            dist = ", ".join(f"{k}={v}" for k, v in sorted(pump_counts.items()))
            print(f"✓ Training data distribution (all files used): {dist}")

        return all_files

    def _load_files(self, data_path, file_list):
        """Load, clean, one-hot encode, smooth, and concatenate a set of files."""
        df_list = []
        for filename in file_list:
            file_path = os.path.join(data_path, filename)
            df = pd.read_csv(file_path)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            if "pump_id" in df.columns:
                df = self.create_dummies(df)

            df_filled = df.ffill(limit=1)
            df_filled = df_filled.dropna()

            if len(df_filled) < 5:
                print(
                    f"  ⚠ Skipping {filename}: insufficient data after cleaning ({len(df_filled)} rows)"
                )
                continue

            df_filtered = self.filter_savitzky_golay(df_filled)
            df_list.append(df_filtered)

        if not df_list:
            raise ValueError("No valid files found in the provided file list")

        return pd.concat(df_list)

    def get_data(self, data_path, is_training=False):
        """Convenience wrapper to list and load all files for a path."""
        file_list = self._get_file_list(data_path, is_training)
        return self._load_files(data_path, file_list)

    def create_dummies(self, df):
        """One-hot encode pump IDs 1..4."""
        pump_ids = [1, 2, 3, 4]
        for pid in pump_ids:
            df[f"pump_id_{pid}"] = (df["pump_id"] == pid).astype(int)

        df = df.drop(columns=["pump_id"], errors="ignore")
        return df

    def filter_savitzky_golay(self, df, window_length=5, polyorder=2):
        """Apply Savitzky-Golay smoothing to non-pump_id columns."""
        df_filtered = df.copy()
        for col in df.columns:
            if "pump_id" not in col:
                df_filtered[col] = savgol_filter(
                    df[col], window_length=window_length, polyorder=polyorder
                )

        return df_filtered

    def rebuild_pump_id(self, df):
        """Rebuild integer pump_id from one-hot columns."""
        pump_cols = [col for col in df.columns if col.startswith("pump_id_")]
        df["pump_id"] = (
            df[pump_cols].idxmax(axis=1).str.split("_").str[-1].astype(int)
        )
        return df

    def build_dataset(self, train=True):
        """
        Build train/val/test datasets with file-level splitting.

        train=True returns:
            (x_train, y_train, ts_train, pids_train,
             x_val_normal, y_val_normal, ts_val_normal, pids_val_normal,
             x_test_normal, y_test_normal, ts_test_normal, pids_test_normal,
             x_val_abnormal, y_val_abnormal, ts_val_abnormal, pids_val_abnormal,
             x_test_abnormal, y_test_abnormal, ts_test_abnormal, pids_test_abnormal)

        train=False returns:
            (x_test, y_test, timestamps, pump_ids)
        """
        if train:
            all_files = self._get_file_list(self.hparams.train_path, is_training=True)

            train_files, remaining = train_test_split(
                all_files,
                train_size=self.train_split,
                random_state=self.seed,
            )

            val_frac = self.val_normal_split / (1 - self.train_split)
            val_files, test_files = train_test_split(
                remaining,
                train_size=val_frac,
                random_state=self.seed,
            )

            print("\n✓ Normal data split (file-level — no leakage):")
            print(f"  Training files:   {len(train_files)} ({self.train_split*100:.0f}%)")
            print(
                f"  Validation files: {len(val_files)} ({self.val_normal_split*100:.0f}%)"
            )
            print(
                f"  Test files:       {len(test_files)} ({self.test_normal_split*100:.0f}%)"
            )

            df_train = self._load_files(self.hparams.train_path, train_files)

            norm_path = getattr(self.hparams, "norm_path", "norm_params.json")
            norm_params = self.get_normalization_params(df_train, save_path=norm_path)

            train_normalized = self.normalize_data(
                df_train, norm_params, self.hparams.norm_method
            )
            train_normalized = self.rebuild_pump_id(train_normalized)
            x_train, y_train, ts_train, pids_train = self.build_preprocessing_window(
                train_normalized, self.hparams.past_history
            )

            df_val = self._load_files(self.hparams.train_path, val_files)
            val_normalized = self.normalize_data(df_val, norm_params, self.hparams.norm_method)
            val_normalized = self.rebuild_pump_id(val_normalized)
            x_val_normal, y_val_normal, ts_val_normal, pids_val_normal = (
                self.build_preprocessing_window(val_normalized, self.hparams.past_history)
            )

            df_test_n = self._load_files(self.hparams.train_path, test_files)
            test_n_normalized = self.normalize_data(
                df_test_n, norm_params, self.hparams.norm_method
            )
            test_n_normalized = self.rebuild_pump_id(test_n_normalized)
            x_test_normal, y_test_normal, ts_test_normal, pids_test_normal = (
                self.build_preprocessing_window(test_n_normalized, self.hparams.past_history)
            )

            all_abnormal_files = self._get_file_list(
                self.hparams.test_path, is_training=False
            )
            val_abnormal_files, test_abnormal_files = train_test_split(
                all_abnormal_files,
                train_size=self.abnormal_val_split,
                random_state=self.seed,
            )

            print("\n✓ Abnormal data split (file-level):")
            print(
                f"  Validation files: {len(val_abnormal_files)} ({self.abnormal_val_split*100:.0f}%)"
            )
            print(
                "  Test files:       "
                f"{len(test_abnormal_files)} ({(1-self.abnormal_val_split)*100:.0f}%)"
            )

            df_val_abn = self._load_files(self.hparams.test_path, val_abnormal_files)
            val_abn_normalized = self.normalize_data(
                df_val_abn, norm_params, self.hparams.norm_method
            )
            val_abn_normalized = self.rebuild_pump_id(val_abn_normalized)
            x_val_abnormal, y_val_abnormal, ts_val_abnormal, pids_val_abnormal = (
                self.build_preprocessing_window(val_abn_normalized, self.hparams.past_history)
            )

            df_test_abn = self._load_files(self.hparams.test_path, test_abnormal_files)
            test_abn_normalized = self.normalize_data(
                df_test_abn, norm_params, self.hparams.norm_method
            )
            test_abn_normalized = self.rebuild_pump_id(test_abn_normalized)
            x_test_abnormal, y_test_abnormal, ts_test_abnormal, pids_test_abnormal = (
                self.build_preprocessing_window(test_abn_normalized, self.hparams.past_history)
            )

            # Safety squeeze in case downstream code ever returns [N, 1, n_output]
            if y_train.ndim == 3 and y_train.shape[1] == 1:
                y_train = np.squeeze(y_train, axis=1)
            if y_val_normal.ndim == 3 and y_val_normal.shape[1] == 1:
                y_val_normal = np.squeeze(y_val_normal, axis=1)
            if y_test_normal.ndim == 3 and y_test_normal.shape[1] == 1:
                y_test_normal = np.squeeze(y_test_normal, axis=1)
            if y_val_abnormal.ndim == 3 and y_val_abnormal.shape[1] == 1:
                y_val_abnormal = np.squeeze(y_val_abnormal, axis=1)
            if y_test_abnormal.ndim == 3 and y_test_abnormal.shape[1] == 1:
                y_test_abnormal = np.squeeze(y_test_abnormal, axis=1)

            ts_train = np.array(ts_train)
            pids_train = np.array(pids_train)
            ts_val_normal = np.array(ts_val_normal)
            pids_val_normal = np.array(pids_val_normal)
            ts_test_normal = np.array(ts_test_normal)
            pids_test_normal = np.array(pids_test_normal)
            ts_val_abnormal = np.array(ts_val_abnormal)
            pids_val_abnormal = np.array(pids_val_abnormal)
            ts_test_abnormal = np.array(ts_test_abnormal)
            pids_test_abnormal = np.array(pids_test_abnormal)

            print("\n✓ Windowed dataset shapes:")
            print(f"  x_train:        {x_train.shape} | y_train:        {y_train.shape}")
            print(
                f"  x_val_normal:   {x_val_normal.shape} | y_val_normal:   {y_val_normal.shape}"
            )
            print(
                f"  x_test_normal:  {x_test_normal.shape} | y_test_normal:  {y_test_normal.shape}"
            )
            print(
                f"  x_val_abnormal: {x_val_abnormal.shape} | y_val_abnormal: {y_val_abnormal.shape}"
            )
            print(
                f"  x_test_abnormal:{x_test_abnormal.shape} | y_test_abnormal:{y_test_abnormal.shape}"
            )

            return (
                x_train,
                y_train,
                ts_train,
                pids_train,
                x_val_normal,
                y_val_normal,
                ts_val_normal,
                pids_val_normal,
                x_test_normal,
                y_test_normal,
                ts_test_normal,
                pids_test_normal,
                x_val_abnormal,
                y_val_abnormal,
                ts_val_abnormal,
                pids_val_abnormal,
                x_test_abnormal,
                y_test_abnormal,
                ts_test_abnormal,
                pids_test_abnormal,
            )

        df_test = self.get_data(self.hparams.test_path, is_training=False)

        with open(self.hparams.norm_path, "r") as file:
            norm_params = json.load(file)

        test = self.normalize_data(df_test, norm_params, self.hparams.norm_method)
        test = self.rebuild_pump_id(test)
        x_test, y_test, timestamps, pump_ids = self.build_preprocessing_window(
            test, self.hparams.past_history
        )

        if y_test.ndim == 3 and y_test.shape[1] == 1:
            y_test = np.squeeze(y_test, axis=1)

        return x_test, y_test, np.array(timestamps), np.array(pump_ids)

    def build_preprocessing_window(self, data, past_history):
        """
        Build conditional-regressor windows grouped by (pump_id, date).

        Input window: past_history rows of input_variables  -> [past_history, n_input]
        Output:       current row of output_variables       -> [n_output]
        """
        input_vars = self.hparams.input_variables
        output_vars = self.hparams.output_variables

        X, Y = [], []
        timestamps = []
        pump_ids = []

        data = data.copy()
        data["date"] = data.index.date
        unique_pairs = data[["pump_id", "date"]].drop_duplicates()
        unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
        data = data.drop(columns=["date"])

        for pump_id, date in unique_pairs_list:
            filtered_df = data[
                (data["pump_id"] == pump_id) & (data.index.date == date)
            ]
            filtered_df = filtered_df.drop(columns=["pump_id"])

            if len(filtered_df) >= 1:
                input_vals = filtered_df[input_vars].values
                for i in range(len(filtered_df)):
                    if i < past_history - 1:
                        n_pad = past_history - 1 - i
                        pad = np.tile(input_vals[0], (n_pad, 1))
                        input_window = np.vstack([pad, input_vals[: i + 1]])
                    else:
                        input_window = input_vals[i - past_history + 1 : i + 1]
                    output_window = filtered_df[output_vars].iloc[i].values

                    X.append(input_window)
                    Y.append(output_window)
                    timestamps.append(filtered_df.index[i])
                    pump_ids.append(pump_id)

        return np.array(X), np.array(Y), timestamps, pump_ids
