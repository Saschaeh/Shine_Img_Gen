import os
import time
import requests
from PIL import Image, ImageEnhance
from io import BytesIO
import re
import logging
import numpy as np
import torch
from rembg import remove
import tempfile  # Added for temp file handling
import zipfile  # Added for zipping outputs
import streamlit as st  # Added for the web UI

try:
    import torch_directml
    directml_available = torch_directml.is_available()
except ImportError:
    directml_available = False
    torch_directml = None

# Configuration (your original config here)
API_KEYS = []  # We'll load from st.secrets later
NUM_IMAGES_PER_VIEW = 2
VIEWS = ["front view", "side left view", "side right view", "back view"]
STANDARD_SIZE = (800, 800)
BACKGROUND_COLOR = (255, 255, 255)  # Target white background
OUTPUT_DIR = 'output'
CANDIDATE_IMAGES = 20  # Increased to capture more matches
LOGO_PATH = 'logo.png'  # Path to your PNG logo file
LOGO_SIZE = (100, 100)  # Logo size
PADDING = 10  # Padding from the edge
LOGO_POSITION = 'bottom_right'  # Options: 'bottom_right', 'top_left', 'bottom_left', 'top_right'
MIN_IMAGE_RES = (400, 400)  # Minimum resolution to accept raw image
UPSCALE_THRESHOLD = (600, 600)  # Upscale only if below this resolution
UPSCALE_FACTOR = 1.5  # Subtle upscaling factor
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
TIMEOUT = 10  # Timeout for API requests in seconds
WHITE_THRESHOLD = 245  # Stricter RGB value threshold for white background

# Your functions here (copy-paste all: is_white_background, sanitize_filename, get_serapi_results, scrape_images, download_image, is_image_valid, upscale_image, standardize_image)

# Setup directories and logging (your original setup)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)

# Detect device (your original)
device = torch_directml.device() if directml_available else torch.device("cpu")
logging.info(f"Using device: {device}")

# Load logo (your original)
try:
    logo = Image.open(LOGO_PATH).convert('RGBA')  # Ensure RGBA mode
    logo = logo.resize(LOGO_SIZE, Image.Resampling.LANCZOS)
    if logo.mode != 'RGBA':
        logging.warning(f"Logo mode is {logo.mode}, converting to RGBA")
        logo = logo.convert('RGBA')
    logging.info("Logo loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load logo: {e}")
    logo = None

# Main processing function (wrap your original main loop)
def process_instrument(asset_id, instrument):
    output_images = []  # Collect paths here
    logging.info(f"Processing: {instrument} (Asset ID: {asset_id})")
    
    # Use temp dir for outputs (ephemeral on web)
    with tempfile.TemporaryDirectory() as temp_dir:
        instrument_dir = os.path.join(temp_dir, asset_id)
        os.makedirs(instrument_dir, exist_ok=True)
        
        used_urls = set()  # Track URLs already used
        sanitized_instrument = sanitize_filename(instrument)
        
        # Check if all views already exist (skip this check for web, as it's always fresh)
        
        for view in VIEWS:
            query = f"{instrument} high resolution product photo {view}"
            urls = scrape_images(query)
            
            saved_for_view = 0
            for url in urls:
                if url in used_urls:
                    logging.warning(f"Skipping duplicate: {url}")
                    continue
                raw_image = download_image(url)
                if raw_image:
                    standardized = standardize_image(raw_image)
                    if standardized:
                        used_urls.add(url)
                        output_path = os.path.join(instrument_dir, f"{sanitized_instrument}_{view.replace(' ', '_')}_{saved_for_view + 1}.jpg")
                        standardized.save(output_path)
                        logging.info(f"Saved: {output_path} (Source: {url.split('?')[0]})")
                        output_images.append(output_path)  # Collect for return
                        saved_for_view += 1
                        if saved_for_view >= NUM_IMAGES_PER_VIEW:
                            break
            
            # Fallback if not enough images (your original fallback)
            if saved_for_view < NUM_IMAGES_PER_VIEW:
                fallback_query = f"{instrument} high resolution product image {view}"
                fallback_urls = scrape_images(fallback_query)
                for url in fallback_urls:
                    if url in used_urls:
                        logging.warning(f"Skipping duplicate (fallback): {url}")
                        continue
                    raw_image = download_image(url)
                    if raw_image:
                        standardized = standardize_image(raw_image)
                        if standardized:
                            used_urls.add(url)
                            output_path = os.path.join(instrument_dir, f"{sanitized_instrument}_{view.replace(' ', '_')}_{saved_for_view + 1}.jpg")
                            standardized.save(output_path)
                            logging.info(f"Saved (fallback): {output_path} (Source: {url.split('?')[0]})")
                            output_images.append(output_path)  # Collect
                            saved_for_view += 1
                            if saved_for_view >= NUM_IMAGES_PER_VIEW:
                                break
            time.sleep(2)  # Delay to respect API rate limits
    
    return output_images

# Streamlit UI (super simple, no auth)
st.title("Instrument Image Processor")
asset_id = st.text_input("Asset ID", value="20")
instrument = st.text_input("Instrument Name", value="Alhambra 1C Spanish Guitar")

# Load API keys securely from Streamlit secrets
API_KEYS = st.secrets.get("api_keys", [])  # Will be set in deployment

if st.button("Process Images"):
    if not API_KEYS:
        st.error("API keys not configured. Add them in app settings.")
    else:
        with st.spinner("Fetching and processing images..."):
            try:
                output_images = process_instrument(asset_id, instrument)
                if output_images:
                    st.success(f"Processed {len(output_images)} images!")
                    for img_path in output_images:
                        st.image(img_path, caption="Processed Image")
                    # Zip and provide download
                    zip_path = f"{asset_id}_images.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for img in output_images:
                            zipf.write(img, arcname=os.path.basename(img))
                    with open(zip_path, 'rb') as f:
                        st.download_button("Download All Images", data=f, file_name=zip_path)
                    os.remove(zip_path)  # Clean up
                else:
                    st.error("No images processed. Check logs or try different input.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logging.error(f"Processing error: {e}")