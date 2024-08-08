import numpy as np
import pandas as pd
import os
import random
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def Mag2Flux(df):
    df_copy = df.dropna().copy()
    df_copy['flux'] = 10 ** (-0.4 * (df_copy['mag'] - 23.9))
    df_copy['flux_error'] = (df_copy['magerr'] / (2.5 / np.log(10))) * df_copy['flux']

    df_copy = df_copy[['obj_id', 'mjd', 'flux', 'flux_error', 'filter', 'type', 'jd']]
    return df_copy

def Normalize_mjd(df):
    df_copy = df.copy()
    df_copy['mjd'] = df_copy.groupby('obj_id')['mjd'].transform(lambda x: x - np.min(x))

    df_copy.reset_index(drop=True, inplace=True)
    return df_copy

def print_types(df, columns=['type_step1', 'type_step2', 'type_step3a', 'type_step3b']):
    for col in columns:
        print(df[col].value_counts())


def minmax_scale(dataframe, scale_columns):
    scaler = MinMaxScaler()
    scaler = scaler.fit(dataframe[scale_columns].to_numpy())
    dataframe.loc[:, scale_columns] = scaler.transform(
        dataframe[scale_columns].to_numpy()
    )
    return dataframe

def create_df_from_file(data_bts_path, data_dir_path):
    df_bts = pd.read_csv(data_bts_path)
    data_files = [f for f in os.listdir(data_dir_path) if f.endswith('.npy')]
    data_names = [f.split('_')[0] for f in data_files]

    data = pd.DataFrame(data_names, columns=['name'])
    data['file'] = data_files
    data = data.merge(df_bts[['obj_id', 'type_step1', 'type_step2', 'type_step3a', 'type_step3b']],
                      left_on='name', right_on='obj_id', how='left')

    data = data.drop(columns=['obj_id'])
    data = data.sort_values(by='file')
    data = data.reset_index(drop=True)
    return data

def split_and_compute_class_weights(df, label_col, split_ratio=0.8, random_seed=42, nb=None, verbose=False):
    types_dict = {
        'type_step1': ['Other', 'SN'],
        'type_step2': ['SN I', 'SN II'],
        'type_step3a': ['SN Ia', 'SN Ib/c'],
        'type_step3b': ['SN II', 'SN IIn/b']
    }

    if nb is not None:
        df = df.groupby(label_col).head(nb)

    train_df_list, val_df_list = [], []
    unique_labels = df[label_col].unique()

    for label in unique_labels:
        df_filtered = df[df[label_col] == label]
        unique_obj_ids = df_filtered['name'].unique()
        random.seed(random_seed)
        random.shuffle(unique_obj_ids)
        split_idx = int(len(unique_obj_ids) * split_ratio)
        train_obj_ids = unique_obj_ids[:split_idx]
        val_obj_ids = unique_obj_ids[split_idx:]
        train_df_list.append(df_filtered[df_filtered['name'].isin(train_obj_ids)])
        val_df_list.append(df_filtered[df_filtered['name'].isin(val_obj_ids)])

    train_df = pd.concat(train_df_list).reset_index(drop=True)
    val_df = pd.concat(val_df_list).reset_index(drop=True)

    train_obj_ids = train_df['name'].unique()
    val_obj_ids = val_df['name'].unique()

    assert len(set(train_obj_ids).intersection(set(val_obj_ids))) == 0

    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_df[label_col])
    class_weight_dict = dict(zip(unique_labels, class_weights))
    #class_weights_mapped = {0: class_weight_dict['Other'], 1: class_weight_dict['SN']}
    class_weights_mapped = {0: class_weight_dict[types_dict[label_col][0]], 1: class_weight_dict[types_dict[label_col][1]]}

    train_files = train_df['file'].tolist()
    val_files = val_df['file'].tolist()

    if verbose:
        print_types(train_df, columns=[label_col])
        print_types(val_df, columns=[label_col])

    return train_files, val_files, class_weights_mapped

def get_data(df, n=20):
    step_3a_df = df[df['type_step3a'] == 0]
    step_3a_df = step_3a_df.sample(n=n, random_state=42)

    step_3ab_df = df[df['type_step3a'] == 1]
    step_3ab_df = step_3ab_df.sample(n=n, random_state=42)

    step_3b_df = df[df['type_step3b'] == 0]
    step_3b_df = step_3b_df.sample(n=n, random_state=42)

    step_3bb_df = df[df['type_step3b'] == 1]
    step_3bb_df = step_3bb_df.sample(n=n, random_state=42)

    other_df = df[df['type_step1'] == 0]
    other_df = other_df.sample(n=n*4, random_state=42)

    res_df = pd.concat([step_3a_df, step_3ab_df, step_3b_df, step_3bb_df, other_df])
    return res_df