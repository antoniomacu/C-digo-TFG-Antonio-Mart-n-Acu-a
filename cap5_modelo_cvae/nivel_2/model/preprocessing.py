# Preprocessing
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from argparse import Namespace

from scipy.signal import savgol_filter
import random
from sklearn.model_selection import train_test_split

class Preprocessor():

    def __init__(self, parameters, seed=42, 
                 train_split=0.8, val_normal_split=0.1, abnormal_val_split=0.5):
        """
        Args:
            parameters: Hyperparameters configuration
            seed: Random seed for reproducibility
            train_split: Fraction of normal data for training (0.8 = 80%)
            val_normal_split: Fraction of normal data for validation (0.1 = 10%)
            abnormal_val_split: Fraction of abnormal data for validation (rest goes to test)
        """
        self.hparams = parameters   # Stores hyperparameters configuration in self.hparams
        self.seed = seed  # Random seed for reproducibility
        self.train_split = train_split  # Fraction of normal data for training (80%)
        self.val_normal_split = val_normal_split  # Fraction of normal data for validation (10%)
        self.test_normal_split = 1 - train_split - val_normal_split  # Remaining 10% for test
        self.abnormal_val_split = abnormal_val_split  # Fraction of abnormal data for validation
        self.norm_params_cached: dict | None = None

    def normalize_data(self, data, params, method):
        """
        Normalizes the given DataFrame based on the provided normalization parameters.
        It uses z-score / min-max scaling (depending)
        Args:
            data (pd.DataFrame): The data to normalize.
            params (dict): Normalization parameters for each feature.
            method (str): The normalization method to apply ('zscore' or 'min-max').
            
        Returns:
            pd.DataFrame: The normalized data.
        """
        
        normalized_data = pd.DataFrame(index=data.index)  # Inicializa un DataFrame vacío para los datos normalizados

        for col in data.columns:
            if col not in params:
                raise Exception(f"Los parámetros de normalización para la columna '{col}' no están disponibles.")
            
            if method == 'zscore':
                mean = params[col]["mean"]
                std = params[col]["std"]

                normalized_data[col] = (data[col] - mean) / std
            
            elif method == 'min-max':
                min_ = params[col]["min"]
                max_ = params[col]["max"]

                normalized_data[col] = (data[col] - min_) / (max_ - min_)
            
            else:
                raise Exception("El método proporcionado no es válido.")
        
        return normalized_data
    
    def denormalize_data(self, data_normalized, params, method):
        """Denormalizes the given DataFrame based on the provided normalization parameters.
        
        Args:
            data_normalized (pd.DataFrame): The normalized data to denormalize.
            params (dict): Normalization parameters for each feature.
            method (str): The normalization method used ('zscore' or 'min-max').
            
        Returns:
            pd.DataFrame: The denormalized data.
        """
        
        denormalized_data = pd.DataFrame(index=data_normalized.index)  # Inicializa un DataFrame vacío para los datos denormalizados
        for col in data_normalized.columns:
            if col not in params:
                raise Exception(f"Los parámetros de denormalización para la columna '{col}' no están disponibles.")
            
            if method == 'zscore':
                mean = params[col]["mean"]
                std = params[col]["std"]
                # Desnormaliza usando la media y la desviación estándar
                denormalized_data[col] = data_normalized[col] * std + mean
                
            elif method == 'min-max':
                min_ = params[col]["min"]
                max_ = params[col]["max"]
                # Desnormaliza usando el rango min-max

                denormalized_data[col] = data_normalized[col] * (max_ - min_) + min_
                        
            else:
                raise Exception("El método proporcionado no es válido.")
            
        return denormalized_data

    def get_normalization_params(self, data, save_path='norm_params.json'):
        """Calculates the normalization parameters for the given data.
            (Parameters for each feature)
        
        Args:
            data: DataFrame with training data
            save_path: Path where to save the norm_params JSON file
        """
        # Inicializa un diccionario para almacenar los parámetros de normalización por variable
        norm_params = {}
        
        for col in data.columns:
            # Calcula los parámetros para cada variable
            mean = float(np.mean(data[col]))
            std = float(np.std(data[col]))
            min_ = float(np.min(data[col]))
            max_ = float(np.max(data[col]))
            
            # Si la columna comienza con 'pump_id', establece su máximo a 1.0
            if col.startswith('pump_id'):
                max_ = 1.0 
            
            # Almacena los parámetros en el diccionario usando el nombre de la columna como clave
            norm_params[col] = {
                "mean": mean,
                "std": std,
                "min": min_,
                "max": max_
            }

        # Guarda los parámetros en un archivo JSON
        self.norm_params_cached = norm_params

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w') as archivo:
            json.dump(norm_params, archivo, indent=4)

        return norm_params        

    def _get_file_list(self, data_path, is_training=False):
        """Get the list of CSV files to process.
        
        All files are loaded — pump balancing is handled at the DataLoader
        level via WeightedRandomSampler (no data is discarded).
        """
        all_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
        
        if is_training:
            # Count per pump for logging only
            from collections import Counter
            pump_counts = Counter(
                f.split('_2')[0] for f in all_files  # e.g. "pump_1", "pump_3"
            )
            dist = ', '.join(f"{k}={v}" for k, v in sorted(pump_counts.items()))
            print(f"✓ Training data distribution (all files used): {dist}")
        
        return all_files

    def _load_files(self, data_path, file_list):
        """Load, clean, and smooth a specific list of CSV files.
        
        This is the inner loop previously inside get_data(), now separated
        so it can be called on each file-level split independently.
        """
        df_list = []
        for filename in file_list:
            file_path = os.path.join(data_path, filename)
            df = pd.read_csv(file_path)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            if 'pump_id' in df.columns:
                df = self.create_dummies(df)

            df_filled = df.ffill(limit=1)
            df_filled = df_filled.dropna()
            
            if len(df_filled) < 5:
                print(f"  ⚠ Skipping {filename}: insufficient data after cleaning ({len(df_filled)} rows)")
                continue
            
            df_filtered = self.filter_savitzky_golay(df_filled)
            df_list.append(df_filtered)

        if not df_list:
            raise ValueError("No valid files found in the provided file list")
        
        return pd.concat(df_list)

    def get_data(self, data_path, is_training=False):
        """Convenience wrapper: list files then load them all."""
        file_list = self._get_file_list(data_path, is_training)
        return self._load_files(data_path, file_list)

    def create_dummies(self, df):
        pump_ids = [1, 2, 3, 4]  # Set of pump IDs to one-hot encode (adapted to current data)
        
        # Vectorised: compare the whole column at once instead of iterating row by row
        for pid in pump_ids:
            df[f'pump_id_{pid}'] = (df['pump_id'] == pid).astype(int)

        df = df.drop(columns=['pump_id'], errors='ignore') 
        
        return df

    def filter_savitzky_golay(self, df, window_length=5, polyorder=2): 
        """
        Apply filter for smoothing
        Params:
        window_length=5:
        The filter uses a window of 5 consecutive data points to fit the polynomial. This means each smoothed value is calculated using itself and its 2 neighbors on each side. 
        The window length must be odd and should be large enough to smooth noise but small enough to preserve features.
        polyorder=2:
        The filter fits a 2nd-degree polynomial (a quadratic curve) to the data in each window. This allows it to capture and preserve trends and curves in the data, rather than 
        just straight lines
        """
        df_filtered = df.copy()
        for col in df.columns: 
            if 'pump_id' not in col:    # Smooths all sensor data columns except the IDs
                df_filtered[col] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
        
        return df_filtered
            
    def rebuild_pump_id(self, df):
        # Vectorised: find which pump_id_X column is 1 for each row, extract X
        pump_cols = [c for c in df.columns if c.startswith('pump_id_')]
        # idxmax returns the column name with the highest value per row (the one that is 1)
        # Then split 'pump_id_X' and grab the number
        df['pump_id'] = (
            df[pump_cols]
            .idxmax(axis=1)
            .str.split('_')
            .str[-1]
            .astype(int)
        )
        
        return df

    def build_dataset(self, train=True):
        """
        Builds the dataset for training, validation, and testing.
        
        IMPORTANT: Splits happen at the FILE level (each CSV = one pump-day).
        This prevents data leakage — no day can appear in two splits.
        
        For training (train=True):
            - Lists files in train_path, splits FILES into 80/10/10
            - Loads each file group independently
            - Normalisation params computed from training files only
            - Abnormal files (test_path) split 50/50 at file level too
        
        For testing only (train=False):
            - Loads all abnormal data from test_path
        """
        
        if train:
            # ===== GET FILE LIST (with pump_3 subsampling) =====
            all_train_files = self._get_file_list(self.hparams.train_path, is_training=True)
            
            # ===== SPLIT FILES — entire days stay together =====
            train_files, temp_files = train_test_split(
                all_train_files,
                test_size=(1 - self.train_split),
                random_state=self.seed
            )
            val_files, test_files = train_test_split(
                temp_files,
                test_size=0.5,
                random_state=self.seed
            )
            
            print(f"\n✓ Normal data split (file-level — no leakage):")
            print(f"  Training files:   {len(train_files)} ({self.train_split*100:.0f}%)")
            print(f"  Validation files: {len(val_files)} ({self.val_normal_split*100:.0f}%)")
            print(f"  Test files:       {len(test_files)} ({self.test_normal_split*100:.0f}%)")
            
            # ===== LOAD AND PROCESS TRAINING DATA =====
            df_train = self._load_files(self.hparams.train_path, train_files)
            norm_params = self.get_normalization_params(df_train)
            
            train_normalized = self.normalize_data(df_train, norm_params, self.hparams.norm_method)
            train_normalized = self.rebuild_pump_id(train_normalized)
            x_train, y_train, ts_train, pids_train = self.build_preprocessing_window(train_normalized, self.hparams.past_history)
            ts_train = np.array(ts_train)
            pids_train = np.array(pids_train)
            
            # ===== LOAD AND PROCESS VALIDATION (NORMAL) DATA =====
            df_val = self._load_files(self.hparams.train_path, val_files)
            val_normalized = self.normalize_data(df_val, norm_params, self.hparams.norm_method)
            val_normalized = self.rebuild_pump_id(val_normalized)
            x_val_normal, y_val_normal, ts_val_normal, pids_val_normal = self.build_preprocessing_window(val_normalized, self.hparams.past_history)
            ts_val_normal = np.array(ts_val_normal)
            pids_val_normal = np.array(pids_val_normal)
            
            # ===== LOAD AND PROCESS TEST (NORMAL) DATA =====
            df_test_n = self._load_files(self.hparams.train_path, test_files)
            test_n_normalized = self.normalize_data(df_test_n, norm_params, self.hparams.norm_method)
            test_n_normalized = self.rebuild_pump_id(test_n_normalized)
            x_test_normal, y_test_normal, ts_test_normal, pids_test_normal = self.build_preprocessing_window(test_n_normalized, self.hparams.past_history)
            ts_test_normal = np.array(ts_test_normal)
            pids_test_normal = np.array(pids_test_normal)
            
            print(f"\n✓ Normal data samples:")
            print(f"  Training:   {x_train.shape[0]} samples")
            print(f"  Validation: {x_val_normal.shape[0]} samples")
            print(f"  Test:       {x_test_normal.shape[0]} samples")
            
            # ===== LOAD AND PROCESS ABNORMAL DATA (file-level split) =====
            all_abnormal_files = self._get_file_list(self.hparams.test_path, is_training=False)
            
            val_abn_files, test_abn_files = train_test_split(
                all_abnormal_files,
                test_size=(1 - self.abnormal_val_split),
                random_state=self.seed
            )
            
            df_val_abn = self._load_files(self.hparams.test_path, val_abn_files)
            val_abn_normalized = self.normalize_data(df_val_abn, norm_params, self.hparams.norm_method)
            val_abn_normalized = self.rebuild_pump_id(val_abn_normalized)
            x_val_abnormal, y_val_abnormal, ts_val_abnormal, pids_val_abnormal = self.build_preprocessing_window(val_abn_normalized, self.hparams.past_history)
            ts_val_abnormal = np.array(ts_val_abnormal)
            pids_val_abnormal = np.array(pids_val_abnormal)
            
            df_test_abn = self._load_files(self.hparams.test_path, test_abn_files)
            test_abn_normalized = self.normalize_data(df_test_abn, norm_params, self.hparams.norm_method)
            test_abn_normalized = self.rebuild_pump_id(test_abn_normalized)
            x_test_abnormal, y_test_abnormal, ts_test_abnormal, pids_test_abnormal = self.build_preprocessing_window(test_abn_normalized, self.hparams.past_history)
            ts_test_abnormal = np.array(ts_test_abnormal)
            pids_test_abnormal = np.array(pids_test_abnormal)
            
            print(f"\n✓ Abnormal data split (file-level):")
            print(f"  Validation files: {len(val_abn_files)} ({self.abnormal_val_split*100:.0f}%)")
            print(f"  Test files:       {len(test_abn_files)} ({(1-self.abnormal_val_split)*100:.0f}%)")
            print(f"  Validation samples: {x_val_abnormal.shape[0]}")
            print(f"  Test samples:       {x_test_abnormal.shape[0]}")
            
            return (x_train, y_train, ts_train, pids_train,
                    x_val_normal, y_val_normal, ts_val_normal, pids_val_normal,
                    x_test_normal, y_test_normal, ts_test_normal, pids_test_normal,
                    x_val_abnormal, y_val_abnormal, ts_val_abnormal, pids_val_abnormal,
                    x_test_abnormal, y_test_abnormal, ts_test_abnormal, pids_test_abnormal)

        else:
            # For standalone testing - load all abnormal data
            df_test = self.get_data(self.hparams.test_path, is_training=False)

            with open(self.hparams.norm_path, 'r') as file:
                norm_params = json.load(file)

            test = self.normalize_data(df_test, norm_params, self.hparams.norm_method)
            test = self.rebuild_pump_id(test)
            x_test, y_test, ts_test, pids_test = self.build_preprocessing_window(test, self.hparams.past_history)
            
            return x_test, y_test, ts_test, pids_test

    def build_preprocessing_window(self, train, past_history): 
        """
        train: preprocessed dataset to be sliced
        past_history: number of time steps to look back (rows)
        
        Returns:
            X: numpy array of input windows
            Y: numpy array of output windows
            timestamps: list of timestamps corresponding to each sample
            pump_ids: list of pump IDs corresponding to each sample
        """
        
        # Get input and output variable names from parameters.json
        input_vars = self.hparams.input_variables 
        output_vars = self.hparams.output_variables

        # Almacenar las ventanas de entrada (X) y salida (Y)
        X, Y = [], []
        timestamps = []  # Store timestamps for each sample
        pump_ids = []    # Store pump IDs for each sample

        train['date'] = train.index.date # convert the date to the index
        unique_pairs = train[['pump_id', 'date']].drop_duplicates() # unique values 

        # convert the unique_pairs into a list of tuples, each tuple is (pump_id, date) i.e: (2, datetime.date(2024,1,23))
        unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None)) 
        train = train.drop(columns=['date'])

        for pump_id, date in unique_pairs_list:
            filtered_df = train[(train['pump_id'] == pump_id) & (train.index.date == date)] # filter by pump_id and date
            filtered_df = filtered_df.drop(columns=['pump_id']) # drop pump_id column for window creation
            if len(filtered_df) > past_history: # Safety check - ensure that the current pump ran the number of time steps required (past_history) (0 - past_history)
                for start in range(past_history, len(filtered_df)): # starts at index past_history(i.e 5) and moves one step at a time until the end of the data
                    # grabs slice start (start - past_history +1) and slice end (start + 1). It creates a sequence depending in past history (if past_history = 1 grabs rows 1,2..,5)
                    input_window = filtered_df[input_vars].iloc[start - past_history+1:start+1].values 
                    # Full VAE (Option B): output = input (reconstruct what was received)
                    output_window = filtered_df[output_vars].iloc[start - past_history+1:start+1].values
                    
                    X.append(input_window)
                    Y.append(output_window)
                    timestamps.append(filtered_df.index[start])  # Capture the timestamp
                    pump_ids.append(pump_id)  # Capture the pump ID

        return np.array(X), np.array(Y), timestamps, pump_ids