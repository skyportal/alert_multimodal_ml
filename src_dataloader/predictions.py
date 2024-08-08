import os
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import src_dataloader.plot_data as plot_data
import random
import src_dataloader.model as T2
import src_dataloader.alert_processor as AlertProcessor
import src_dataloader.tools as tools

def load_data(file_path, step):
    sample = np.load(file_path, allow_pickle=True).item()
    X = (sample['photometry'], sample['metadata'].to_numpy(), sample['images'])
    y = sample[step]
    return X, y

def get_obj_id(file):
    return file.split('_')[0]

def group_files_by_obj_id(test_dir):
    files = os.listdir(test_dir)
    obj_id_to_files = {}
    for file in files:
        obj_id = get_obj_id(file)
        if obj_id not in obj_id_to_files:
            obj_id_to_files[obj_id] = []
        obj_id_to_files[obj_id].append(file)
    for obj_id in obj_id_to_files:
        obj_id_to_files[obj_id] = sorted(obj_id_to_files[obj_id], key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
    return obj_id_to_files

def predict_alerts(files, model, preprocessed_path, step):
    step_type = 'type_' + step
    data = [load_data(os.path.join(preprocessed_path, file), step_type) for file in files]
    data = [d for d in data if d is not None]
    if not data:
        return [], []
    batch_x, batch_y = zip(*data)
    batch_x = [np.array([item[i] for item in batch_x]) for i in range(len(batch_x[0]))]
    predictions = model.predict(batch_x, verbose=0)
    ground_truths = np.array(batch_y)
    return predictions, ground_truths, data

import plotly.graph_objects as go

def plot_interactive_supernova_classification(df, types):
    fig = go.Figure()

    hover_texts = [
        f"Obj ID: {row['obj_id']}<br>"
        f"Alerte: {row['alert_num']}<br>"
        f"Prediction: {round(row['prediction']*100, 2)}%<br>"
        f"Class Prediction: {types[row['class_prediction']]}<br>"
        f"Ground Truth: {types[row['ground_truth']]}"
        for _, row in df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=df['alert_num'],
        y=df['prediction'],
        fill='tozeroy',
        mode='none',
        name=types[1],
        hoverinfo='text',
        text=hover_texts,
        fillcolor='orange'
    ))

    fig.add_trace(go.Scatter(
        x=df['alert_num'],
        y=[1] * len(df),
        fill='tonexty',
        mode='none',
        name=types[0],
        hoverinfo='none',
        fillcolor='blue'
    ))

    fig.update_layout(
        title='Supernova Classification Over Time',
        xaxis_title='Alertes',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        width=1200,
        height=600   
    )

    fig.show()

from tensorflow.keras.optimizers import Adam

def define_model():
    input_shapes = [(200, 4), (10,), (63, 63, 3)]
    
    num_filters = 64
    num_classes = 1
    num_layers = 1
    d_model = 64
    num_heads = 16
    dff = 128
    rate = 0.4

    model_instance = T2.T2Model(num_filters=num_filters, num_classes=num_classes, num_layers=num_layers,
                                   d_model=d_model, num_heads=num_heads, dff=dff, input_shapes=input_shapes, 
                                   rate=rate)
    
    model_instance.build(input_shapes)

    model_instance.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model_instance

def plot_interactive_supernova_classification_mjd(df, types):
    fig = go.Figure()

    hover_texts = [
        f"Obj ID: {row['obj_id']}<br>"
        f"MJD: {row['mjd']}<br>"
        f"Prediction: {round(row['prediction']*100, 2)}%<br>"
        f"Class Prediction: {types[row['class_prediction']]}<br>"
        f"Ground Truth: {types[row['ground_truth']]}"
        for _, row in df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=df['mjd'],
        y=df['prediction'],
        fill='tozeroy',
        mode='none',
        name=types[1],
        hoverinfo='text',
        text=hover_texts,
        fillcolor='orange'
    ))

    fig.add_trace(go.Scatter(
        x=df['mjd'],
        y=[1] * len(df),
        fill='tonexty',
        mode='none',
        name=types[0],
        hoverinfo='none',
        fillcolor='blue'
    ))

    fig.update_layout(
        title='Supernova Classification Over Time',
        xaxis_title='MJD',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        width=1200,
        height=600
    )

    fig.show()

from tqdm import tqdm

def supernova_classification(test_data_dir, step, num_predictions=None, shuffle=True, verbose=True):
    types_dict = {
        'step1': ['Other', 'SN'],
        'step2': ['SN I', 'SN II'],
        'step3a': ['SN Ia', 'SN Ib/c'],
        'step3b': ['SN II', 'SN IIn/b']
    }

    weight_files = {
        'step1': 'checkpoints/step1_V1.weights.h5',
        'step2': 'checkpoints/step1_V1.weights.h5',
        'step3a': 'checkpoints/step3a_V1.weights.h5',
        'step3b': 'checkpoints/step3b_V1.weights.h5'
    }

    model = define_model()
    
    model.load_weights(weight_files[step])  
      
    obj_id_to_files = group_files_by_obj_id(test_data_dir)

    if shuffle:
        obj_id_list = list(obj_id_to_files.keys())
        random.shuffle(obj_id_list)
        obj_id_to_files = {obj_id: obj_id_to_files[obj_id] for obj_id in obj_id_list}
    
    all_results = []
    summary_results = [] 
    types = types_dict[step]
    
    obj_id_list = list(obj_id_to_files.keys())
    if num_predictions is not None:
        obj_id_list = obj_id_list[:num_predictions]

    for obj_id in tqdm(obj_id_list, desc='Obj ID', unit='obj_id'):
        files = obj_id_to_files[obj_id]
        obj_results = []
        predictions, ground_truths, data = predict_alerts(files, model, test_data_dir, step)
        
        metadata_df, images = AlertProcessor.AlertProcessor.get_process_alerts(obj_id, 'data_kowalski')
        metadata_df = metadata_df.sort_values(by='jd')
        metadata_df = metadata_df.reset_index(drop=True)
        metadata_df['mjd'] = metadata_df['jd'] - 2400000.5
        metadata_df = tools.Normalize_mjd(metadata_df)

        for i in range(len(predictions)):
            obj_results.append({
                'obj_id': obj_id,
                'alert_num': i + 1,
                'mjd': metadata_df['mjd'].iloc[i],
                'prediction': predictions[i][0],
                'class_prediction': (predictions[i][0] > 0.5).astype(int),
                'ground_truth': ground_truths[i]
            })

        final_prediction = np.mean([pred['prediction'] for pred in obj_results])
        final_class = (final_prediction > 0.5).astype(int)
        confidence = final_prediction if final_class == 1 else 1 - final_prediction
        ground_truth = obj_results[0]['ground_truth']

        if verbose:
            print(f"Obj ID: {obj_id}")
            print(f"Ground Truth: {types[ground_truth]}")
            print(f"Final Prediction: {types[final_class]}")
            print(f"Confidence: {confidence*100:.2f}%")

        summary_results.append({
            'obj_id': obj_id,
            'ground_truth': types[ground_truth],
            'final_prediction': types[final_class],
            'confidence': confidence * 100
        })
        
        df_obj = pd.DataFrame(obj_results)
        all_results.extend(obj_results)

        if verbose:
            plot_interactive_supernova_classification(df_obj, types)
            plot_interactive_supernova_classification_mjd(df_obj, types)

            max_alert_num = max([res['alert_num'] for res in obj_results])
            plot_data.plot_photometry(obj_id, alerte=max_alert_num)

    df_all_results = pd.DataFrame(all_results)
    df_summary_results = pd.DataFrame(summary_results)
    
    return df_all_results, df_summary_results