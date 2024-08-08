import os
import numpy as np
import multiprocessing
import src_dataloader.gaussian_process as gp
import src_dataloader.tools as tools
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocessor:
    @staticmethod
    def process_gp(photo_df):
        df_gp_ready = tools.Mag2Flux(photo_df)
        df_gp_ready = tools.Normalize_mjd(df_gp_ready).drop_duplicates().reset_index(drop=True)
        return df_gp_ready

    @staticmethod
    def cut_photometry(photo_df, metadata_df, index, max_mjd=200):
        jd_current = metadata_df['jd'].iloc[index]
        photometry_filtered = photo_df[photo_df['jd'] <= jd_current]
        return None if photometry_filtered['mjd'].max() > max_mjd else photometry_filtered

    @staticmethod
    def preprocess_metadata(metadata_df):
        metadata_df = metadata_df.drop_duplicates(subset=['jd'], keep='first')
        columns_metadata = [
            "sgscore1", "sgscore2", 
            "distpsnr1", "distpsnr2", 
            "ra", 
            "dec", 
            #"ndethist", 
            "nmtchps", 
            #"ncovhist", 
            "sharpnr", 
            "scorr", 
            "sky",
            'jd'
        ]
        return metadata_df[columns_metadata].fillna(-999.0)

    @staticmethod
    def get_num_workers(reserved_cpus=5):
        return max(1, multiprocessing.cpu_count() - reserved_cpus)
    
def remove_filter(photo_df):
    filters = photo_df['filter'].unique()
    for filt in filters:
        if len(photo_df[photo_df['filter'] == filt]) < 3:
            photo_df = photo_df[photo_df['filter'] != filt]
    photo_df = photo_df.reset_index(drop=True)
    return photo_df

def create_heatap(data):
    mjd = data['mjd'].values
    flux_ztfg = data['flux_ztfg'].values
    flux_ztfi = data['flux_ztfi'].values
    flux_ztfr = data['flux_ztfr'].values

    flux_error_ztfg = data['flux_error_ztfg'].values
    flux_error_ztfi = data['flux_error_ztfi'].values
    flux_error_ztfr = data['flux_error_ztfr'].values

    pb_wavelengths = {
        'ztfg': 4800.,
        'ztfr': 6400.,
        'ztfi': 7900.,
    }

    # Créer les valeurs de la grille pour le temps et la longueur d'onde
    unique_mjd = np.unique(mjd)
    wavelengths = np.array([pb_wavelengths['ztfg'], pb_wavelengths['ztfr'], pb_wavelengths['ztfi']])

    grid_time, grid_wavelength = np.meshgrid(unique_mjd, wavelengths)

    # Organiser les flux et les incertitudes dans des grilles
    grid_flux = np.vstack([flux_ztfg, flux_ztfr, flux_ztfi])
    grid_flux_error = np.vstack([flux_error_ztfg, flux_error_ztfr, flux_error_ztfi])

    # Normaliser les prédictions de flux
    grid_flux_normalized = grid_flux / np.nanmax(grid_flux)

    grid_flux_error_norm = grid_flux_error / np.nanmax(grid_flux_error)

    images = np.stack((grid_flux_normalized, grid_flux_error_norm), axis=0)

    return images
    
def process_and_save_sample(args):
    res_dict = {}
    sample, save_dir, kernel, is_i_filter = args
    obj_id = sample['obj_id']
    alerte = sample['alerte']

    save_path = os.path.join(save_dir, f"{obj_id}_alert_{alerte}.npy")
    if os.path.exists(save_path):
        return

    gp_ready = sample['photometry']

    gp_ready = remove_filter(gp_ready)

    if len(gp_ready) == 0:
        return

    gp_final = gp.process_gaussian(gp_ready, kernel=kernel, number_gp=200)


    columns = ['flux_ztfg', 'flux_error_ztfg', 'flux_ztfr', 'flux_error_ztfr']
    if is_i_filter:
        columns += ['flux_ztfi', 'flux_error_ztfi']
    
    for col in columns:
        if col not in gp_final.columns:
            gp_final[col] = 0.
            if not is_i_filter:
                return

    # gp_final = tools.minmax_scale(gp_final, columns)

    # useful_columns = ['mjd', 'flux_ztfg', 'flux_ztfr']
    # if is_i_filter:
    #     useful_columns += ['flux_ztfi']
    # sequences = gp_final[useful_columns].values
    # padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

    res_dict.update({
        'obj_id': obj_id,
        'photometry': create_heatap(gp_final),#padded_sequences,
        'metadata': sample['metadata'],
        'images': sample['images'],
        'type_step1': sample['type_step1'],
        'type_step2': sample['type_step2'],
        'type_step3a': sample['type_step3a'],
        'type_step3b': sample['type_step3b'],
        'target': sample['target'],
        'alerte': alerte
    })

    np.save(save_path, res_dict)