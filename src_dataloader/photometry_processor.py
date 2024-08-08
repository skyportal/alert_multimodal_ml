import pandas as pd
import os

class PhotometryProcessor:
    @staticmethod
    def clean_photometry(df, df_type, is_i_filter=True):
        df = PhotometryProcessor.clean_dataframe(df, is_i_filter)
        df['type'] = df_type[df_type['obj_id'] == df['obj_id'].iloc[0]]['type'].values[0]
        df.dropna(subset=['mag', 'magerr'], inplace=True)
        return df.reset_index(drop=True)
    
    @staticmethod
    def clean_dataframe(df, is_i_filter=True):
        df = df.rename(columns={
            'magpsf': 'mag',
            'sigmapsf': 'magerr',
            'fid': 'filter',
            'scorr': 'snr',
            'diffmaglim': 'limiting_mag'
        })
        df['filter'] = df['filter'].replace({1: 'ztfg', 2: 'ztfr', 3: 'ztfi'})
        if not is_i_filter:
            df = df[df['filter'] != 'ztfi']
        df['mjd'] = df['jd'] - 2400000.5
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter']]
        return df


    @staticmethod
    def process_csv(name, df_bts, base_path, is_i_filter=True):
        file_path = os.path.join(base_path, name, 'photometry.csv')
        return PhotometryProcessor.clean_photometry(pd.read_csv(file_path), df_bts, is_i_filter) if os.path.exists(file_path) else pd.DataFrame()

    @staticmethod
    def get_first_valid_index(df, min_points=3):
        filter_counts = {'ztfr': 0, 'ztfg': 0, 'ztfi': 0}
        for i in range(len(df)):
            current_filter = df['filter'].iloc[i]
            if current_filter in filter_counts:
                filter_counts[current_filter] += 1
                if filter_counts[current_filter] >= min_points:
                    return i
        return -1

    @staticmethod
    def add_metadata_to_photometry(photo_df, metadata_df, is_i_filter=True):
        metadata_df_copy = PhotometryProcessor.clean_dataframe(metadata_df.copy(), is_i_filter)
        df = pd.merge(photo_df, metadata_df_copy, on=['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter'], how='outer', suffixes=('', '_metadata'))        
        df = df[['obj_id', 'jd', 'mjd', 'mag', 'magerr', 'snr', 'limiting_mag', 'filter', 'type']]
        df['obj_id'] = df['obj_id'].ffill().bfill()
        df['type'] = df['type'].ffill().bfill()
        df = df.drop_duplicates(subset=['mjd', 'filter'], keep='first')
        df = df.sort_values(by=['mjd'])
        df.reset_index(drop=True, inplace=True)
        return df
