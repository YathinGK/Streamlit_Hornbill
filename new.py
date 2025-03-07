import os
import shutil
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define Directories
INPUT_DIR = "Uploaded_Images"  # Folder where uploaded images will be stored
OUTPUT_DIR = "organized_images"
CATEGORY_DIR = os.path.join(OUTPUT_DIR, "Classified_Images")
DATE_DIR = os.path.join(OUTPUT_DIR, "Date_Wise_Images")
MODEL_PATH = "Trained_Model/thermal_visible_classifier.h5"
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CATEGORY_DIR, exist_ok=True)
os.makedirs(DATE_DIR, exist_ok=True)

# Load trained AI model
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Error: Model file '{MODEL_PATH}' not found! Train the model first.")
    st.stop()

model = load_model(MODEL_PATH)

# Function to extract metadata
def extract_metadata(image_path):
    metadata = {"FileName": os.path.basename(image_path)}
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_data = {GPSTAGS.get(t, t): v for t, v in value.items()}
                    lat, lon = extract_gps_coordinates(gps_data)
                    metadata["Latitude"] = lat
                    metadata["Longitude"] = lon
                elif tag_name == "DateTime":
                    date_part, time_part = format_datetime(value)
                    metadata["Date"] = date_part
                    metadata["Time"] = time_part
    except Exception as e:
        st.warning(f"Error extracting metadata from {image_path}: {e}")
    return metadata

# Function to extract GPS coordinates
def extract_gps_coordinates(gps_data):
    try:
        lat_values = gps_data.get("GPSLatitude", (0, 0, 0))
        lon_values = gps_data.get("GPSLongitude", (0, 0, 0))
        lat = lat_values[0] + (lat_values[1] / 60.0) + (lat_values[2] / 3600.0)
        lon = lon_values[0] + (lon_values[1] / 60.0) + (lon_values[2] / 3600.0)
        return f"{lat:.10f}", f"{lon:.10f}"
    except:
        return "Unknown", "Unknown"

# Function to format datetime
def format_datetime(datetime_str):
    try:
        date_part, time_part = datetime_str.split(" ")
        formatted_date = date_part.replace(":", "-")
        return formatted_date, time_part
    except:
        return "Unknown", "Unknown"

# Function to classify image as Thermal or Visible
def predict_image_category(image_path):
    img_height, img_width = 224, 224
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Visible" if prediction[0][0] > 0.5 else "Thermal"

# Streamlit UI
st.title("📸 AI-Based Image Classification & Metadata Extraction")
st.write("Upload images to classify them as Thermal or Visible and extract metadata.")

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    metadata_list = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(INPUT_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process Image
        metadata = extract_metadata(file_path)
        date = metadata.get("Date", "Unknown")
        time = metadata.get("Time", "Unknown")
        lat = metadata.get("Latitude", "Unknown")
        lon = metadata.get("Longitude", "Unknown")
        new_filename = f"{date}_{lat}_{lon}.jpg" if date != "Unknown" else f"unknown_{uploaded_file.name}"
        category = predict_image_category(file_path)
        
        # Organize images
        category_folder = os.path.join(CATEGORY_DIR, category)
        os.makedirs(category_folder, exist_ok=True)
        date_folder = os.path.join(DATE_DIR, date if date != "Unknown" else "Unknown_Date")
        os.makedirs(date_folder, exist_ok=True)
        category_path = os.path.join(category_folder, new_filename)
        date_path = os.path.join(date_folder, new_filename)
        shutil.copy(file_path, category_path)
        shutil.move(file_path, date_path)
        
        # Append metadata
        metadata_list.append({
            "Original FileName": uploaded_file.name,
            "New FileName": new_filename,
            "Date": date,
            "Time": time,
            "Latitude": lat,
            "Longitude": lon,
            "Category": category,
            "Category Path": category_path,
            "Date Path": date_path
        })

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(METADATA_CSV, index=False)
    st.success("✅ Images processed and organized successfully!")
    st.download_button("📥 Download Metadata CSV", data=metadata_df.to_csv(index=False), file_name="metadata.csv", mime="text/csv")
    
    # Display organized images
    st.subheader("📂 Organized Images")
    for category in ["Thermal", "Visible"]:
        st.write(f"### {category} Images")
        category_path = os.path.join(CATEGORY_DIR, category)
        images = [f for f in os.listdir(category_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        if images:
            st.image([os.path.join(category_path, img) for img in images], caption=images, width=150)
        else:
            st.write("No images found.")
