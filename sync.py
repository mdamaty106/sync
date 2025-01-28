import streamlit as st
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import io
import pandas as pd

# -----------------------------------------------------------
# 1) Streamlit Title
# -----------------------------------------------------------
st.title("Sync in 3 Clicks")

# -----------------------------------------------------------
# 2) Two File Uploads Only
# -----------------------------------------------------------
uploaded_lidar_zip = st.file_uploader("Upload Velodyne Data", type="zip")
uploaded_image_zip = st.file_uploader("Upload Image Data", type="zip")

# Create temporary folders
lidar_temp_dir = "lidar_temp"
image_temp_dir = "image_temp"
os.makedirs(lidar_temp_dir, exist_ok=True)
os.makedirs(image_temp_dir, exist_ok=True)

# -----------------------------------------------------------
# Utility: Unzip to folder
# -----------------------------------------------------------
def unzip_to_folder(uploaded_file, out_dir):
    """Unzip an uploaded file into a specified directory."""
    zip_path = os.path.join(out_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

# -----------------------------------------------------------
# Utility: Find LiDAR .txt Files
# -----------------------------------------------------------
def find_velodyne_txt_files(base_dir):
    txt_files = []
    for root, dirs, files in os.walk(base_dir):
        if "velodyne_points" in root and "data" in root:
            for file in files:
                if file.endswith(".txt") and not file.startswith("._"):
                    txt_files.append(os.path.join(root, file))
    return sorted(txt_files)

# -----------------------------------------------------------
# Utility: Find image files
# -----------------------------------------------------------
def find_image_files(base_dir):
    img_files = []
    for root, dirs, files in os.walk(base_dir):
        if "image_02" in root and "data" in root:
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")) and not file.startswith("._"):
                    img_files.append(os.path.join(root, file))
    return sorted(img_files)

# -----------------------------------------------------------
# Utility: Load LiDAR from a .txt
# -----------------------------------------------------------
def load_lidar_from_txt(file_path):
    return np.loadtxt(file_path, delimiter=' ')

# -----------------------------------------------------------
# 3D LiDAR Visualization
# -----------------------------------------------------------
def plot_lidar_3d(lidar_data):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        lidar_data[:, 0],
        lidar_data[:, 1],
        lidar_data[:, 2],
        c='blue', s=1, alpha=0.5
    )
    ax.set_title("3D LiDAR Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    st.pyplot(fig)

# -----------------------------------------------------------
# Utility: Synchronize LiDAR and Image Data
# -----------------------------------------------------------
def synchronize_data(lidar_files, image_files):
    synced_pairs = []
    lidar_base_names = {os.path.splitext(os.path.basename(f))[0]: f for f in lidar_files}
    image_base_names = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    common_keys = sorted(set(lidar_base_names.keys()) & set(image_base_names.keys()))
    for key in common_keys:
        synced_pairs.append((lidar_base_names[key], image_base_names[key]))
    return synced_pairs

# -----------------------------------------------------------
# After Upload: Unzip & Synchronize Data
# -----------------------------------------------------------
lidar_files = []
image_files = []
synced_data = []

if uploaded_lidar_zip:
    unzip_to_folder(uploaded_lidar_zip, lidar_temp_dir)
    lidar_files = find_velodyne_txt_files(lidar_temp_dir)

if uploaded_image_zip:
    unzip_to_folder(uploaded_image_zip, image_temp_dir)
    image_files = find_image_files(image_temp_dir)

if lidar_files and image_files:
    synced_data = synchronize_data(lidar_files, image_files)
    st.success(f"Found {len(synced_data)} synchronized pairs.")
elif uploaded_lidar_zip or uploaded_image_zip:
    st.warning("Please upload both LiDAR and image data to synchronize.")

# -----------------------------------------------------------
# UI: View Synchronized Data
# -----------------------------------------------------------
if synced_data:
    st.markdown("---")
    st.subheader("View Synced Data")
    selected_pair = st.selectbox("Select a synchronized pair:", synced_data, format_func=lambda x: f"LiDAR: {os.path.basename(x[0])}, Image: {os.path.basename(x[1])}")

    if st.button("Visualize Synchronized Pair"):
        try:
            lidar_data = load_lidar_from_txt(selected_pair[0])
            st.write(f"LiDAR File: {os.path.basename(selected_pair[0])}")
            columns = ["x", "y", "z", "intensity"]
            df_preview = pd.DataFrame(lidar_data[:5], columns=columns)
            st.write("First 5 LiDAR points:")
            st.dataframe(df_preview)
            plot_lidar_3d(lidar_data)

            image = cv2.imread(selected_pair[1])[:, :, ::-1]
            st.write(f"Image File: {os.path.basename(selected_pair[1])}")
            st.image(image, caption="Synchronized Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error visualizing synchronized data: {e}")
else:
    st.info("Upload and synchronize data to visualize.")

# -----------------------------------------------------------
# UI: Export Synchronized Data
# -----------------------------------------------------------
st.markdown("---")
st.subheader("Export Synced Data")

export_format = st.selectbox("Select export format:", [".zip", ".h5"])

if st.button("Export"):
    if not synced_data:
        st.error("No synchronized data available to export.")
    else:
        try:
            if export_format == ".zip":
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for lidar_file, image_file in synced_data:
                        zf.write(lidar_file, os.path.basename(lidar_file))
                        zf.write(image_file, os.path.basename(image_file))
                zip_buffer.seek(0)
                st.download_button("Download ZIP", data=zip_buffer, file_name="synced_data.zip", mime="application/zip")

            elif export_format == ".h5":
                h5_buffer = io.BytesIO()
                with pd.HDFStore(h5_buffer, mode="w") as h5_store:
                    for i, (lidar_file, image_file) in enumerate(synced_data):
                        lidar_data = load_lidar_from_txt(lidar_file)
                        image_data = cv2.imread(image_file)
                        h5_store.put(f"pair_{i}/lidar", pd.DataFrame(lidar_data, columns=["x", "y", "z", "intensity"]))
                        h5_store.put(f"pair_{i}/image", pd.DataFrame(image_data.reshape(-1, 3), columns=["R", "G", "B"]))
                h5_buffer.seek(0)
                st.download_button("Download H5", data=h5_buffer, file_name="synced_data.h5", mime="application/octet-stream")

        except Exception as e:
            st.error(f"Error exporting data: {e}")





