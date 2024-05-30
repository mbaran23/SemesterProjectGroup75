import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import io
import base64
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import json

def fetch_latest_data(dev_eui):
    url = "url"

    current_time = datetime.now(timezone.utc)
    from_time = current_time - timedelta(hours=2)
    
    payload = json.dumps({
        "dev_eui": dev_eui,
        "from": from_time.strftime('%Y-%m-%d %H:%M:%S.%f'), 
        "to": current_time.strftime('%Y-%m-%d %H:%M:%S.%f') 
    })
    
    username = "username"
    pwd = "password"
    basic_auth = base64.b64encode(f"{username}:{pwd}".encode())
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Basic {basic_auth.decode("utf-8")}'
    }
    
    response = requests.request("GET", url, headers=headers, data=payload)
    if response.status_code != 200:
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return None
    
    csv_data = response.text
    df = pd.read_csv(io.StringIO(csv_data))
    return df

def preprocess_and_identify(df, model):
    df['magnitude'] = (df['x']**2 + df['y']**2 + df['z']**2)**0.5
    df = df.fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler()
    df[['x', 'y', 'z', 'temperature', 'magnitude']] = scaler.fit_transform(df[['x', 'y', 'z', 'temperature', 'magnitude']])
    
    predictions = model.predict(df[['x', 'y', 'z', 'temperature', 'magnitude']])
    df['is_anomaly'] = predictions
    df['is_anomaly'] = df['is_anomaly'].map({1: 'Spot Taken', 0: 'Spot Free'})
    return df

# Define available models
models = {
    'Spot50': '/Users/magdalenabaran/Desktop/Project_Group_75/pipelines/spot50_k_nn/spot50_k_nn.pkl',
    'Spot58': '/Users/magdalenabaran/Desktop/Project_Group_75/pipelines/spot58_k_nn/spot58_k_nn.pkl',
    'Spot59': '/Users/magdalenabaran/Desktop/Project_Group_75/pipelines/spot59_k_nn/spot59_k_nn.pkl',
    'Spot63': '/Users/magdalenabaran/Desktop/Project_Group_75/pipelines/spot63_k_nn/spot63_k_nn.pkl',
    'Spot6e': '/Users/magdalenabaran/Desktop/Project_Group_75/pipelines/spot6e_k_nn/spot6e_k_nn.pkl'
}

st.title("Parking Spot Availability Check")
model_selection = st.selectbox('Select a model:', list(models.keys()))
model_path = models[model_selection]
model = joblib.load(model_path)

spot_selection = st.selectbox('Choose a parking spot:', ['70B3D56380000163', '70B3D56380000159', '70B3D56380000158', '70B3D56380000150', '70B3D5638000016E'])

if st.button('Check Availability'):
    df = fetch_latest_data(spot_selection)
    if df is not None and not df.empty:
        results = preprocess_and_identify(df, model)
        results['timestamp'] = pd.to_datetime(results['time']) + timedelta(hours=1)  
        results = results[['timestamp', 'is_anomaly']].tail(2)  
        st.write(results)
