import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Global variables
df = None
scaler = None
model = None
boundary_lat_min = None
boundary_lat_max = None
boundary_lon_min = None
boundary_lon_max = None

# Function to create dummy data
def create_dummy_data():
    data = {
        'Time': pd.to_datetime(pd.to_datetime('2025-07-05 14:00:00') + pd.to_timedelta(np.arange(500), unit='m')),
        'Latitude': np.random.uniform(7.805242, 7.806483, 500),
        'Longitude': np.random.uniform(5.494747, 5.495450, 500)
    }
    df_dummy = pd.DataFrame(data)
    df_dummy['Time'] = df_dummy['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df_dummy

# Data loading function
def load_data(uploaded_file):
    global df
    if uploaded_file is None:
        st.write("No file uploaded. Using dummy data.")
        df = create_dummy_data()
    else:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded file loaded successfully.")
    
    # Ensure the expected columns are present
    expected_columns = ['Time', 'Latitude', 'Longitude']
    if not all(col in df.columns for col in expected_columns):
        st.error(f"CSV must contain columns: {expected_columns}")
        return None
    
    # Convert Time column to datetime
    try:
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        if df['Time'].isna().any():
            st.warning("Some timestamps failed to parse with '%Y-%m-%d %H:%M:%S'. Attempting mixed format parsing.")
            df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')
        if df['Time'].isna().any():
            st.error("Some timestamps could not be parsed. Please ensure all Time values are valid.")
            return None
    except Exception as e:
        st.error(f"Error parsing Time column: {e}")
        return None
    
    # Handle missing values
    df = df.dropna(subset=['Latitude', 'Longitude'])
    if df.empty:
        st.error("No valid data after dropping missing values.")
        return None
    
    # Sort by Time
    df = df.sort_values(by='Time')
    return df

# Boundary check function
def check_boundary(latitude, longitude, boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max):
    if boundary_lat_min is None or boundary_lat_max is None or boundary_lon_min is None or boundary_lon_max is None:
        return False
    return (latitude < boundary_lat_min or latitude > boundary_lat_max or
            longitude < boundary_lon_min or longitude > boundary_lon_max)

# Data preprocessing for LSTM
def preprocess_data(df, sequence_length):
    global scaler
    if df is None:
        return None, None, None

    features = ['Latitude', 'Longitude']
    df = df.dropna(subset=features)
    if df.empty:
        st.error("No valid data after dropping missing values.")
        return None, None, None

    df = df.sort_values(by='Time')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length), :])
        y.append(scaled_data[i + sequence_length, :2])

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("Not enough data to create sequences with the specified sequence length.")
        return None, None, None

    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape, lstm_units):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dense(2))  # Output Latitude and Longitude
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM model
def train_model(X_train, y_train, X_test, y_test, epochs, batch_size, lstm_units):
    global model
    if X_train is None or y_train is None:
        st.error("Training data is not available.")
        return None

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), lstm_units=lstm_units)
    st.write("Training model...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test) if X_test is not None and y_test is not None else None, verbose=0)
    st.write("Training finished.")
    return history

# Predict next location
def predict_next_location(model, last_sequence, scaler):
    if model is None or last_sequence is None or scaler is None:
        st.error("Model or data not available for prediction.")
        return None

    scaled_last_sequence = scaler.transform(last_sequence)
    scaled_last_sequence = np.reshape(scaled_last_sequence, (1, scaled_last_sequence.shape[0], scaled_last_sequence.shape[1]))
    predicted_scaled = model.predict(scaled_last_sequence)

    dummy_input_for_inverse_transform = np.zeros((predicted_scaled.shape[0], scaler.n_features_in_))
    dummy_input_for_inverse_transform[:, :predicted_scaled.shape[1]] = predicted_scaled
    predicted_original = scaler.inverse_transform(dummy_input_for_inverse_transform)[:, :2]

    return predicted_original[0]

# Visualization functions
def plot_gps_readings(df):
    if df is None or 'Time' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        st.error("Data not available or missing required columns for GPS plot.")
        return

    time_seconds = (df['Time'] - df['Time'].min()).dt.total_seconds()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(time_seconds, df['Latitude'], c='blue', label='Latitude')
    ax.scatter(time_seconds, df['Longitude'], c='red', label='Longitude')
    ax.set_xlabel('Time (seconds since start)')
    ax.set_ylabel('Coordinates (°)')
    ax.set_title('GPS Readings of Tracked Cattle Over Time')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_spatial_gps(df, boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max, predicted_loc=None):
    if df is None or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        st.error("Data not available or missing required columns for spatial GPS plot.")
        return

    colors = ['red' if check_boundary(row['Latitude'], row['Longitude'], boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max) else 'green' for _, row in df.iterrows()]
    lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
    lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()
    lat_range = max(lat_max - lat_min, 0.00001)
    lon_range = max(lon_max - lon_min, 0.00001)
    padding = max(lat_range, lon_range) * 0.1
    x_min, x_max = min(lon_min, boundary_lon_min) - padding, max(lon_max, boundary_lon_max) + padding
    y_min, y_max = min(lat_min, boundary_lat_min) - padding, max(lat_max, boundary_lat_max) + padding

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df['Longitude'], df['Latitude'], c=colors, s=5, alpha=0.6, label='Cattle Position')
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((boundary_lon_min, boundary_lat_min), 
                           boundary_lon_max - boundary_lon_min, 
                           boundary_lat_max - boundary_lat_min,
                           fill=False, edgecolor='blue', linewidth=2, label='Boundary'))
    if predicted_loc is not None:
        ax.scatter(predicted_loc[1], predicted_loc[0], c='purple', s=50, label='Predicted Next')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Spatial GPS Plot with Boundary')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_confusion_matrix(true_labels, predicted_labels):
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))
    cm = np.array([[tn, fp], [fn, tp]])

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Breach', 'Breach'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Predicted No', 'Predicted Yes'])
    ax.set_title('Confusion Matrix for Boundary Detection')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='bottom', fontsize=12)
    plt.colorbar(ax.imshow(cm, cmap='Blues'), label='Count')
    plt.text(-0.4, -0.1, f'Accuracy: {accuracy:.2f}', ha='left', va='center', fontsize=10, transform=ax.transAxes)
    plt.text(-0.4, -0.2, f'Sensitivity: {sensitivity:.2f}', ha='left', va='center', fontsize=10, transform=ax.transAxes)
    plt.text(-0.4, -0.3, f'Specificity: {specificity:.2f}', ha='left', va='center', fontsize=10, transform=ax.transAxes)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Sensitivity: {sensitivity:.2f}")
    st.write(f"Specificity: {specificity:.2f}")

def plot_roc_curve(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve for Boundary Detection')
    ax.legend(loc='lower right')
    ax.grid(True)
    st.pyplot(fig)

    st.write(f"AUC: {roc_auc:.2f}")

def plot_training_loss(history):
    if history is None:
        st.error("No training history available.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='val loss')
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Streamlit app
st.title("Cattle GPS Data Visualizer with LSTM")
st.write("1. Upload your CSV file with columns [Time, Latitude, Longitude] (optional, dummy data will be used if no file is uploaded).")
st.write("2. Set training parameters and train the LSTM model.")
st.write("3. Set boundary coordinates.")
st.write("4. Click 'Visualize Results' or 'Predict Next Location'.")

# UI elements
st.subheader("Upload GPS Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

st.subheader("LSTM Training Parameters")
sequence_length = st.number_input("Sequence Length", min_value=1, max_value=100, value=10)
epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=32)
lstm_units = st.number_input("LSTM Units", min_value=1, max_value=200, value=50)
if st.button("Train Model"):
    df = load_data(uploaded_file)
    if df is not None:
        X, y, scaler = preprocess_data(df, sequence_length)
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            history = train_model(X_train, y_train, X_test, y_test, epochs, batch_size, lstm_units)
            if history:
                plot_training_loss(history)

st.subheader("Set Boundary Coordinates")
boundary_lat_min = st.number_input("Min Latitude", value=7.805242, format="%.6f")
boundary_lat_max = st.number_input("Max Latitude", value=7.806483, format="%.6f")
boundary_lon_min = st.number_input("Min Longitude", value=5.494747, format="%.6f")
boundary_lon_max = st.number_input("Max Longitude", value=5.495450, format="%.6f")

# if st.button("Predict Next Location"):
#     if model is None or df is None or scaler is None:
#         st.error("Model not trained or data not loaded. Please train the model first.")
#     else:
#         sequence_length = int(sequence_length)
#         if len(df) < sequence_length:
#             st.error(f"Not enough data ({len(df)}) to form a sequence of length {sequence_length}.")
#         else:
#             features = ['Latitude', 'Longitude']
#             last_sequence_df = df.tail(sequence_length)
#             last_sequence = last_sequence_df[features].values
#             predicted_loc = predict_next_location(model, last_sequence, scaler)
#             if predicted_loc is not None:
#                 st.write(f"Predicted next location: Latitude={predicted_loc[0]:.6f}, Longitude={predicted_loc[1]:.6f}")
#                 if check_boundary(predicted_loc[0], predicted_loc[1], boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max):
#                     st.write("ALERT: Predicted location is OUTSIDE the set boundary!")
#                 else:
#                     st.write("Predicted location is inside the set boundary.")

if st.button("Visualize Results"):
    df = load_data(uploaded_file)
    if df is None:
        st.error("Failed to load data.")
    else:
        st.write(f"Boundary set: Lat [{boundary_lat_min}, {boundary_lat_max}], Lon [{boundary_lon_min}, {boundary_lon_max}]")
        
        plot_gps_readings(df)
        plot_spatial_gps(df, boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max)
        
        # Generate true labels and scores
        true_labels = []
        scores = []
        lat_center = (boundary_lat_min + boundary_lat_max) / 2
        lon_center = (boundary_lon_min + boundary_lon_max) / 2
        max_distance = np.sqrt((boundary_lat_max - boundary_lat_min)**2 + (boundary_lon_max - boundary_lon_min)**2)

        for _, row in df.iterrows():
            breach = check_boundary(row['Latitude'], row['Longitude'], boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max)
            true_labels.append(1 if breach else 0)
            distance = np.sqrt((row['Latitude'] - lat_center)**2 + (row['Longitude'] - lon_center)**2)
            score = min(distance / max_distance, 1.0) if max_distance > 0 else 0
            scores.append(score)

        true_labels = np.array(true_labels)
        scores = np.array(scores)

        # Use LSTM predictions for future evaluation
        if model is not None and len(df) >= sequence_length:
            last_sequence_df = df.tail(sequence_length)
            last_sequence = last_sequence_df[features].values
            predicted_loc = predict_next_location(model, last_sequence, scaler)
            if predicted_loc is not None:
                predicted_breach = check_boundary(predicted_loc[0], predicted_loc[1], boundary_lat_min, boundary_lat_max, boundary_lon_min, boundary_lon_max)
                true_labels = np.append(true_labels, 1 if predicted_breach else 0)
                predicted_labels = np.append((scores > 0.5).astype(int), 1 if predicted_breach else 0)
                scores = np.append(scores, min(np.sqrt((predicted_loc[0] - lat_center)**2 + (predicted_loc[1] - lon_center)**2) / max_distance, 1.0))

        threshold = 0.5
        predicted_labels = (scores > threshold).astype(int)
        plot_confusion_matrix(true_labels, predicted_labels)
        plot_roc_curve(true_labels, scores)
