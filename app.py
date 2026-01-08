import joblib
import rasterio
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import os


model = joblib.load('isolation_forest_ndvi.pkl')
print("Model loaded successfully!")


TRAINING_FOLDER = 'NDVI_Images'
training_files = sorted([f for f in os.listdir(TRAINING_FOLDER) if f.endswith('.tif')])
print(f"Found {len(training_files)} training files")


def predict_anomaly(ndvi_file):

    print(f"Reading uploaded file: {ndvi_file}")
    with rasterio.open(ndvi_file) as src:
        user_ndvi = src.read(1)
        print(f"User NDVI shape: {user_ndvi.shape}")

    # Stack
    full_stack = [user_ndvi]

    # Only take up to 23 more images (or less if we don't have 24 total)
    num_additional = min(23, len(training_files))
    for i in range(num_additional):
        file_path = os.path.join(TRAINING_FOLDER, training_files[i])
        with rasterio.open(file_path) as src:
            full_stack.append(src.read(1))

    print(f"Total images in stack: {len(full_stack)}")

    ndvi_stack = np.array(full_stack)  # shape: (time, height, width)
    time, height, width = ndvi_stack.shape
    print(f"Stack shape: {ndvi_stack.shape}")

    # Reshape for model
    X_new = ndvi_stack.reshape(time, height * width).T
    print(f"Reshaped for prediction: {X_new.shape}")

    # Predict anomalies
    y_pred = model.predict(X_new)  # -1 = anomaly, 1 = normal
    print(f"Predictions - Unique values: {np.unique(y_pred, return_counts=True)}")

    anomaly_map = y_pred.reshape(height, width)
    return anomaly_map, user_ndvi


def show_anomaly(ndvi_file):
    """Generate and display anomaly map"""
    try:
        anomaly_map, original_ndvi = predict_anomaly(ndvi_file)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im1 = axes[0].imshow(original_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('Original NDVI', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot anomaly map
        anomaly_visual = (anomaly_map + 1) / 2  # Maps -1→0 (anomaly), 1→1 (normal)

        im2 = axes[1].imshow(anomaly_visual, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title('Anomaly Detection\n(Red=Anomaly, Green=Normal)',
                          fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04,
                     label='0=Anomaly, 1=Normal')

        # Calculate statistics
        num_anomalies = np.sum(anomaly_map == -1)
        total_pixels = anomaly_map.size
        pct_anomaly = (num_anomalies / total_pixels) * 100

        fig.suptitle(f'Anomaly Detection Results\n{num_anomalies:,} anomalous pixels ({pct_anomaly:.2f}%)',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save the figure
        output_path = 'anomaly_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved result to {output_path}")
        print(f"Anomaly statistics: {num_anomalies}/{total_pixels} pixels ({pct_anomaly:.2f}%)")

        return output_path

    except Exception as e:
        print(f"Error in show_anomaly: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return an error image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error processing image:\n{str(e)}',
                ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        error_path = 'error.png'
        plt.savefig(error_path)
        plt.close()
        return error_path


# Create Gradio interface
iface = gr.Interface(
    fn=show_anomaly,
    inputs=gr.File(label="Upload NDVI GeoTIFF file (.tif)", file_types=['.tif']),
    outputs=gr.Image(label="Anomaly Detection Result", type="filepath"),
    title="🌿 Vegetation Anomaly Detection",
    description="""
    Upload an NDVI GeoTIFF image to detect vegetation anomalies using Isolation Forest.

    - **Red areas**: Anomalous vegetation patterns
    - **Green areas**: Normal vegetation patterns

    The model uses temporal patterns from 24 months of training data to identify unusual changes.
    """,
    examples=None
)

if __name__ == "__main__":
    print("\nStarting Gradio interface...")
    print("Make sure 'isolation_forest_ndvi.pkl' and 'NDVI_Images/' folder are in the same directory")
    iface.launch(share=False)