import os
import numpy as np
import rasterio
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


folder = 'NDVI_Images'
files = sorted(os.listdir(folder))

ndvi_stack = []

for f in files:
    with rasterio.open(os.path.join(folder, f)) as src:
        ndvi_stack.append(src.read(1))

ndvi_stack = np.array(ndvi_stack)
print("Stack shape:", ndvi_stack.shape)

# ndvi_stack shape: (24, height, width)
time , height, width = ndvi_stack.shape

# rehsaping to pixels time
X = ndvi_stack.reshape(time, height * width).T # shape: (height * width, 24)
print("Shape for model:", X.shape)

model_file = 'isolation_forest_ndvi.pkl'
if os.path.exists(model_file):
    model = joblib.load(model_file)
    print("Model loaded from file")
else:
    model = IsolationForest(contamination=0.02, random_state=1)
    model.fit(X)
    joblib.dump(model, model_file)
    print("Model trained and saved")

# predict anomalies
y_pred = model.predict(X) # -1= anomaly, 1= normal

anomaly_map = y_pred.reshape(height, width)

plt.figure(figsize=(6, 6))
plt.imshow(anomaly_map, cmap="coolwarm")
plt.colorbar(label='Anomaly (-1= Unsual, 1=Normal)')
plt.title("Vegetation Anomaly Map")
plt.show()

plt.imsave('anomaly_map.png', anomaly_map, cmap='coolwarm')

