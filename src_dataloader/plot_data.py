import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.subplots as sp
import src_dataloader.photometry_processor as PhotometryProcessor
import src_dataloader.alert_processor as AlertProcessor
import src_dataloader.data_preprocessor as DataPreprocessor
import src_dataloader.gaussian_process as gp

def plot_types_distributions(data, form='bar', name_col='type'):

    if form not in ['bar', 'pie']:
        print('Invalid form, please choose between bar and pie')
        return
    
    count = data[name_col].value_counts()

    plt.figure(figsize=(10, 6))
    plt.title('Distribution of Types')
    plt.ylabel('Types')
    plt.xlabel('Count')

    if form == 'bar':
        count.plot(kind='bar')
        for i, v in enumerate(count):
            plt.text(i, v, str(v), ha='center', rotation=45, fontsize=10)
    else:
        count.plot(kind='pie', autopct='%1.1f%%', shadow=True)

    plt.show()

def plot_photometry_before(lc, color_dict=None):
    if color_dict is None:
        color_dict = {'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'y',
              'sdssg': 'green', 'sdssr': 'red', 'sdssi': 'y',
              'atlasc': 'cyan', 'atlaso': 'orange',}
        
    if 'mag' in lc.columns:
        col_norm = 'mag'
        col_err = 'magerr'
    elif 'flux' in lc.columns:
        col_norm = 'flux'
        col_err = 'flux_error'
    else:
        print("No magnitude or flux column found")
        return
    
    fig, ax1 = plt.subplots(1, 1, figsize=(9,6))
    ymin, ymax = np.inf, -np.inf

    for f in set(lc['filter']):
        tf = lc[lc['filter'] == f]
        
        tf_det = tf[tf[col_norm] >= 3.]
        tf_ul = tf
        if 'snr' in tf.columns:
            tf_ul = tf[tf['snr'] < 3]

        ax1.errorbar(tf_det['mjd'].values,
                     tf_det[col_norm], yerr=tf_det[col_err],
                     color=color_dict[f], markeredgecolor='k',
                     label=f, marker='o')
        if np.min(tf_det[col_norm]) < ymin:
            ymin = np.min(tf_det[col_norm])
        if np.max(tf_det[col_norm]) > ymax:
            ymax = np.max(tf_det[col_norm])
                     
        if len(tf_ul) != 0:
            if np.min(tf_det[col_norm]) < ymin:
                ymin = np.min(tf_det[col_norm])
            if np.max(tf_det[col_norm]) > ymax:
                ymax = np.max(tf_det[col_norm])
    
    plt.gca()#.invert_yaxis()
    ax1.set_xlabel("MJD", fontsize=18)
    ax1.set_ylabel("Magnitude (AB)", fontsize=18)
    plt.legend()

    ax1.set_title(f"{lc['obj_id'].values[0]} - {lc['type'].values[0]}")
    plt.show()

def plot_image(image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['Science', 'Template', 'Difference']

    for i, ax in enumerate(axes):
        ax.imshow(image[:, :, i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.show()

def plot_gp(raw_df, gp_df, type_obj=None, alerte=None):
    color_dict = {'ztfg': 'rgba(0, 128, 0, 0.2)', 'ztfr': 'rgba(255, 0, 0, 0.2)', 'ztfi': 'rgba(255, 255, 0, 0.2)'}
    line_color_dict = {'ztfg': 'rgba(0, 128, 0, 1.0)', 'ztfr': 'rgba(255, 0, 0, 1.0)', 'ztfi': 'rgba(255, 255, 0, 1.0)'}

    fig = go.Figure()

    for filter_name in raw_df['filter'].unique():
        filter_data = raw_df[raw_df['filter'] == filter_name]
        fig.add_trace(go.Scatter(
            x=filter_data['mjd'],
            y=filter_data['flux'],
            mode='markers',
            name=f'Raw {filter_name}',
            marker=dict(color=line_color_dict[filter_name]),
            hovertemplate=(
                'mjd: %{x}<br>'
                'flux: %{y}<br>'
                'filter: %{text}<br>'
                'obj_id: ' + filter_data['obj_id'].iloc[0] + '<br>'
                'alert_num: %{customdata}<br>'
            ),
            text=filter_data['filter'],
        ))

    for filter_name in ['ztfg', 'ztfr', 'ztfi']:
        flux_col = f'flux_{filter_name}'
        error_col = f'flux_error_{filter_name}'
        
        fig.add_trace(go.Scatter(
            x=gp_df['mjd'],
            y=gp_df[flux_col],
            mode='lines',
            name=f'GP {filter_name}',
            line=dict(color=line_color_dict[filter_name])
        ))
        fig.add_trace(go.Scatter(
            x=gp_df['mjd'],
            y=gp_df[flux_col] + gp_df[error_col],
            mode='lines',
            line=dict(color=line_color_dict[filter_name].replace('1.0', '0')),
            showlegend=False,
            name=f'Upper Bound {filter_name}'
        ))
        fig.add_trace(go.Scatter(
            x=gp_df['mjd'],
            y=gp_df[flux_col] - gp_df[error_col],
            mode='lines',
            line=dict(color=line_color_dict[filter_name].replace('1.0', '0')),
            fill='tonexty',
            fillcolor=color_dict[filter_name],
            showlegend=False,
            name=f'Lower Bound {filter_name}'
        ))

    fig.update_layout(
        xaxis_title='Time (mjd)',
        yaxis_title='Flux',
        legend=dict(orientation='h'),
        width=1200,
        height=600 
    )

    title = f'Light Curve for Object ID: {raw_df["obj_id"].iloc[0]}'

    if type_obj is not None:
        title += f' - Type: {type_obj}'

    if alerte is not None:
        title += f' - Alert: {alerte}'

    fig.update_layout(title=title)

    fig.show()

def plot_photometry(obj_id, alerte=None, type_obj=None):
    df_bts = pd.read_csv('BTS_categorized.csv')
    base_path = 'data_kowalski/'
    photo_df, metadata_df, _ = PhotometryProcessor.PhotometryProcessor.process_csv(obj_id, df_bts, base_path), *AlertProcessor.AlertProcessor.get_process_alerts(obj_id, base_path)
    photo_df, metadata_df = photo_df.sort_values(by='jd'), metadata_df.sort_values(by='jd')
    photo_df = PhotometryProcessor.PhotometryProcessor.add_metadata_to_photometry(photo_df, metadata_df)
    photo_df = DataPreprocessor.DataPreprocessor.process_gp(photo_df)

    max_mjd = min(photo_df['mjd'].max(), 200)
    photo_ready = photo_df[photo_df['mjd'] <= max_mjd]
    metadata_df = metadata_df[metadata_df['jd'] <= photo_ready['jd'].max()]

    # if alerte is None:
    #     alerte = len(metadata_df) - 1

    # start_alert = PhotometryProcessor.PhotometryProcessor.get_first_valid_index(photo_df)

    # if alerte < start_alert:
    #     alerte = start_alert

    #photo_ready = DataPreprocessor.DataPreprocessor.cut_photometry(photo_df, metadata_df, alerte).copy()

    kernel = pickle.load(open('kernel.pkl', 'rb'))
    gp_final = gp.process_gaussian(photo_ready, kernel=kernel, number_gp=200)

    for i, jd in enumerate(metadata_df['jd'], start=1):
        photo_ready.loc[photo_ready['jd'] == jd, 'alert_num'] = i

    if 'flux_ztfi' not in gp_final.columns:
        gp_final['flux_ztfi'] = 0
        gp_final['flux_error_ztfi'] = 0

    if 'flux_ztfg' not in gp_final.columns:
        gp_final['flux_ztfg'] = 0
        gp_final['flux_error_ztfg'] = 0

    if 'flux_ztfr' not in gp_final.columns:
        gp_final['flux_ztfr'] = 0
        gp_final['flux_error_ztfr'] = 0  

    plot_gp(photo_ready, gp_final, type_obj=type_obj, alerte=alerte)
    return photo_ready, gp_final

def plot_history(history):
    if not isinstance(history, dict):
        history = history.history

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=('Model accuracy', 'Model loss'), shared_xaxes=True)

    fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Train Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history['val_accuracy'], mode='lines', name='Validation Accuracy'), row=1, col=1)

    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Train Loss'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'), row=1, col=2)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)

    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=2)

    fig.update_layout(title='Training History', height=500, width=1200, showlegend=True)

    fig.show()