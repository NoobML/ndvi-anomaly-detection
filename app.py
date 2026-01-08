import joblib
import rasterio
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import os

# Load trained model
model = joblib.load('isolation_forest_ndvi.pkl')

# Path to your 24-month training NDVI images
TRAINING_FOLDER = 'NDVI_Images'
training_files = sorted(os.listdir(TRAINING_FOLDER))  # ensure month order

def predict_anomaly(ndvi_file):
    # Read user-uploaded NDVI
    with rasterio.open(ndvi_file) as src:
        user_ndvi = src.read(1)

    # Stack: first the uploaded image, then fill remaining months from training set
    full_stack = [user_ndvi]
    for f in training_files[1:24]:  # take 23 more images to make 24
        with rasterio.open(os.path.join(TRAINING_FOLDER, f)) as src:
            full_stack.append(src.read(1))

    ndvi_stack = np.array(full_stack)  # shape: (24, height, width)
    time, height, width = ndvi_stack.shape

    # Reshape for model
    X_new = ndvi_stack.reshape(time, height * width).T

    # Predict anomalies
    y_pred = model.predict(X_new)  # -1= anomaly, 1= normal
    anomaly_map = y_pred.reshape(height, width)
    return anomaly_map

def show_anomaly(ndvi_file):
    anomaly = predict_anomaly(ndvi_file)
    plt.figure(figsize=(6,6))
    plt.imshow(anomaly, cmap='coolwarm', vmin=-1, vmax=1)
    plt.axis('off')
    plt.title('Vegetation Anomaly Map')
    plt.savefig('anomaly_map1.png')  # save to file
    plt.close()  # close the figure
    return anomaly

iface = gr.Interface(
    fn=show_anomaly,
    inputs=gr.File(file_types=['.tif']),
    outputs='image'
)

iface.launch()
