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
import tempfile
import zipfile
import streamlit as st
import base64

try:
    import torch_directml
    directml_available = torch_directml.is_available()
except ImportError:
    directml_available = False
    torch_directml = None

# Configuration
API_KEYS = []  # Will load from st.secrets
NUM_IMAGES_PER_VIEW = 2
VIEWS = ["front view", "side left view", "side right view", "back view"]
STANDARD_SIZE = (800, 800)
BACKGROUND_COLOR = (255, 255, 255)  # Target white background
OUTPUT_DIR = 'output'
CANDIDATE_IMAGES = 20
LOGO_PATH = 'logo.png'
LOGO_SIZE = (100, 100)
PADDING = 10
LOGO_POSITION = 'bottom_right'
MIN_IMAGE_RES = (400, 400)
UPSCALE_THRESHOLD = (600, 600)
UPSCALE_FACTOR = 1.5
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
TIMEOUT = 10
WHITE_THRESHOLD = 245

# Function to get base64 of image
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Functions (unchanged from original)
def is_white_background(image):
    try:
        width, height = image.size
        samples = []
        top_left = image.crop((0, 0, min(10, width), min(10, height)))
        samples.append(np.mean(np.array(top_left.convert('RGB')), axis=(0, 1)))
        top_right = image.crop((max(0, width-10), 0, width, min(10, height)))
        samples.append(np.mean(np.array(top_right.convert('RGB')), axis=(0, 1)))
        bottom_left = image.crop((0, max(0, height-10), min(10, width), height))
        samples.append(np.mean(np.array(bottom_left.convert('RGB')), axis=(0, 1)))
        mean_color = np.mean(samples, axis=0)
        logging.info(f"Background color mean across edges: {mean_color}")
        return all(c >= WHITE_THRESHOLD for c in mean_color)
    except Exception as e:
        logging.warning(f"Failed to check background color: {e}")
        return False

def sanitize_filename(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def get_serapi_results(params):
    for api_key in API_KEYS:
        params['api_key'] = api_key
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                logging.warning(f"SerpApi error with key {api_key[:5]}...: {data['error']}")
                continue
            logging.info(f"API response status for {params['q']}: {response.status_code} (using key {api_key[:5]}...)")
            logging.info(f"API response data for {params['q']}: {data.get('search_metadata', {})}")
            return data
        except requests.exceptions.HTTPError as he:
            logging.warning(f"HTTP error with key {api_key[:5]}...: {he}")
            continue
        except Exception as e:
            logging.error(f"Exception with key {api_key[:5]}...: {e}")
            continue
    logging.error("All API keys failed to retrieve results.")
    return {}

def scrape_images(query):
    params = {"engine": "google_images", "q": query, "num": CANDIDATE_IMAGES, "tbs": "isz:l"}
    data = get_serapi_results(params)
    if 'images_results' in data:
        urls = [result.get('original') for result in data.get('images_results', []) if 'original' in result]
        logging.info(f"Found {len(urls)} URLs for query: {query}")
        return urls
    else:
        logging.warning(f"No images found for {query}. Response: {data}")
        fallback_query = query.replace("isolated on white background", "").strip()
        if fallback_query != query:
            logging.info(f"Trying fallback query: {fallback_query}")
            params["q"] = fallback_query
            data = get_serapi_results(params)
            if 'images_results' in data:
                urls = [result.get('original') for result in data.get('images_results', []) if 'original' in result]
                logging.info(f"Found {len(urls)} URLs for fallback query: {fallback_query}")
                return urls
        return []

def download_image(url, retries=3):
    if 'media.johnlewiscontent.com' in url:
        logging.warning(f"Skipping known problematic URL: {url}")
        return None
    headers = {'User-Agent': USER_AGENT}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT)
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logging.warning(f"Skipping non-image URL: {url} (content-type: {content_type})")
                return None
            img = Image.open(BytesIO(response.content))
            if img.size[0] < MIN_IMAGE_RES[0] or img.size[1] < MIN_IMAGE_RES[1]:
                logging.warning(f"Skipping low-resolution image: {url} (size: {img.size})")
                return None
            return img
        except Exception as e:
            logging.warning(f"Download attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    logging.error(f"All download attempts failed for {url}")
    return None

def is_image_valid(image):
    if image is None:
        return False
    try:
        img_array = np.array(image.convert('RGB'))
        mean_pixel = img_array.mean()
        if mean_pixel > 240:
            logging.warning("Image is mostly blank or white.")
            return False
        return True
    except Exception as e:
        logging.error(f"Image validation failed: {e}")
        return False

def upscale_image(image, scale=UPSCALE_FACTOR):
    width, height = image.size
    if width >= UPSCALE_THRESHOLD[0] and height >= UPSCALE_THRESHOLD[1]:
        logging.info(f"Skipping upscaling for {width}x{height} image")
        return image
    try:
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        upscaled = torch.nn.functional.interpolate(img_tensor, scale_factor=scale, mode='bicubic')
        upscaled = (upscaled.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu() * 255).numpy().astype(np.uint8)
        return Image.fromarray(upscaled)
    except Exception as e:
        logging.error(f"Upscaling failed: {e}")
        return image

def standardize_image(input_image):
    try:
        if not is_white_background(input_image):
            logging.info("Removing non-white background")
            output = remove(input_image)
            white_canvas = Image.new("RGBA", input_image.size, BACKGROUND_COLOR + (255,))
            if output.mode == 'RGBA':
                white_canvas.paste(output, (0, 0), output)
            else:
                white_canvas.paste(output, (0, 0))
            output = white_canvas.convert('RGB')
        else:
            logging.info("Image already has white background, skipping removal")
            output = input_image.convert('RGB')
        if not is_image_valid(output):
            return None
        
        output = upscale_image(output)
        
        width, height = output.size
        if height > width:
            new_height = STANDARD_SIZE[1]
            new_width = int(width * (new_height / height))
        else:
            new_width = STANDARD_SIZE[0]
            new_height = int(height * (new_width / width))
        output = output.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        new_image = Image.new("RGB", STANDARD_SIZE, BACKGROUND_COLOR)
        offset = ((STANDARD_SIZE[0] - new_width) // 2, (STANDARD_SIZE[1] - new_height) // 2)
        new_image.paste(output, offset)
        
        enhancer = ImageEnhance.Sharpness(new_image)
        new_image = enhancer.enhance(1.2)
        
        if logo is not None:
            if LOGO_POSITION == 'bottom_right':
                logo_position = (STANDARD_SIZE[0] - logo.width - PADDING, STANDARD_SIZE[1] - logo.height - PADDING)
            elif LOGO_POSITION == 'top_left':
                logo_position = (PADDING, PADDING)
            elif LOGO_POSITION == 'bottom_left':
                logo_position = (PADDING, STANDARD_SIZE[1] - logo.height - PADDING)
            elif LOGO_POSITION == 'top_right':
                logo_position = (STANDARD_SIZE[0] - logo.width - PADDING, PADDING)
            else:
                logo_position = (STANDARD_SIZE[0] - logo.width - PADDING, STANDARD_SIZE[1] - logo.height - PADDING)
            try:
                new_image.paste(logo, logo_position, logo.split()[3] if logo.mode == 'RGBA' else logo.convert('RGBA').split()[3])
            except Exception as e:
                logging.warning(f"Failed to add logo due to transparency issue: {e}")
        
        return new_image
    except Exception as e:
        logging.error(f"Standardization failed: {e}")
        return None

# Setup directories and logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('image_processing.log'), logging.StreamHandler()]
)

device = torch_directml.device() if directml_available else torch.device("cpu")
logging.info(f"Using device: {device}")

try:
    logo = Image.open(LOGO_PATH).convert('RGBA')
    logo = logo.resize(LOGO_SIZE, Image.Resampling.LANCZOS)
    if logo.mode != 'RGBA':
        logging.warning(f"Logo mode is {logo.mode}, converting to RGBA")
        logo = logo.convert('RGBA')
    logging.info("Logo loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load logo: {e}")
    logo = None

# Main processing function - Updated for conditional processing
def process_instrument(asset_id, instrument, uploaded_images=None, edit_uploaded=True, scrap_images=True):
    output_pil_images = []  # PIL for display and ZIP
    file_names = []  # For custom naming
    source_tags = []  # Track source (uploaded or scraped)
    logging.info(f"Processing: {instrument} (Asset ID: {asset_id})")
    
    used_urls = set()
    sanitized_instrument = sanitize_filename(instrument)
    
    # Process uploaded images if selected and provided
    if edit_uploaded and uploaded_images:
        for i, uploaded_file in enumerate(uploaded_images):
            if uploaded_file.type in ['image/jpeg', 'image/png']:
                try:
                    img = Image.open(uploaded_file).convert('RGB')
                    standardized = standardize_image(img)
                    if standardized:
                        view = VIEWS[i % len(VIEWS)]  # Cycle through views
                        file_name = f"{sanitized_instrument}_{view.replace(' ', '_')}_1.jpg"
                        logging.info(f"Processed uploaded: {file_name}")
                        output_pil_images.append(standardized)
                        file_names.append(file_name)
                        source_tags.append("Uploaded")
                except Exception as e:
                    logging.error(f"Failed to process uploaded image {uploaded_file.name}: {e}")
        if len(output_pil_images) >= len(VIEWS) * NUM_IMAGES_PER_VIEW and not scrap_images:
            return output_pil_images, file_names, source_tags
    
    # Fall back to SerpApi if selected and not enough images from uploads
    if scrap_images:
        for view in VIEWS:
            if len(output_pil_images) >= len(VIEWS) * NUM_IMAGES_PER_VIEW:
                break
            query = f"{instrument} high resolution product photo {view}"
            urls = scrape_images(query)
            
            saved_for_view = 0
            for url in urls:
                if url in used_urls or len(output_pil_images) >= len(VIEWS) * NUM_IMAGES_PER_VIEW:
                    continue
                raw_image = download_image(url)
                if raw_image:
                    standardized = standardize_image(raw_image)
                    if standardized:
                        used_urls.add(url)
                        file_name = f"{sanitized_instrument}_{view.replace(' ', '_')}_{saved_for_view + 1}.jpg"
                        logging.info(f"Processed: {file_name} (Source: {url.split('?')[0]})")
                        output_pil_images.append(standardized)
                        file_names.append(file_name)
                        source_tags.append("Scraped")
                        saved_for_view += 1
                        if saved_for_view >= NUM_IMAGES_PER_VIEW:
                            break
            
            if saved_for_view < NUM_IMAGES_PER_VIEW:
                fallback_query = f"{instrument} high resolution product image {view}"
                fallback_urls = scrape_images(fallback_query)
                for url in fallback_urls:
                    if url in used_urls or len(output_pil_images) >= len(VIEWS) * NUM_IMAGES_PER_VIEW:
                        continue
                    raw_image = download_image(url)
                    if raw_image:
                        standardized = standardize_image(raw_image)
                        if standardized:
                            used_urls.add(url)
                            file_name = f"{sanitized_instrument}_{view.replace(' ', '_')}_{saved_for_view + 1}.jpg"
                            logging.info(f"Processed (fallback): {file_name} (Source: {url.split('?')[0]})")
                            output_pil_images.append(standardized)
                            file_names.append(file_name)
                            source_tags.append("Scraped")
                            saved_for_view += 1
                            if saved_for_view >= NUM_IMAGES_PER_VIEW:
                                break
            time.sleep(2)
    
    return output_pil_images, file_names, source_tags

# Function to create ZIP from PIL images
def create_zip_from_images(pil_images, file_names, asset_id):
    """Create a ZIP file from PIL images and return the file content"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pil_img, file_name in zip(pil_images, file_names):
            # Save PIL image to a BytesIO and use writestr
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            zipf.writestr(file_name, img_buffer.read())
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit UI
# Add background image with overlay
bg_img = get_base64_of_bin_file('BG.jpg')  # Use relative path for deployment
css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    position: relative;
    background-image: url("data:image/jpeg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.4);
    z-index: 1;
}}
[data-testid="stAppViewContainer"] > * {{
    position: relative;
    z-index: 2;
    color: white !important;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.title("Instrument Image Processor")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'asset_id' not in st.session_state:
    st.session_state.asset_id = "20"
if 'instrument' not in st.session_state:
    st.session_state.instrument = "Alhambra 1C Spanish Guitar"
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'edit_uploaded' not in st.session_state:
    st.session_state.edit_uploaded = True
if 'scrap_images' not in st.session_state:
    st.session_state.scrap_images = True

# Input fields and checkboxes
asset_id = st.text_input("Asset ID", value=st.session_state.asset_id, key="asset_id_input")
instrument = st.text_input("Instrument Name", value=st.session_state.instrument, key="instrument_input")
uploaded_files = st.file_uploader("Upload your own images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader")

edit_uploaded = st.checkbox("Edit Uploaded Images", value=st.session_state.edit_uploaded, key="edit_uploaded_checkbox")
scrap_images = st.checkbox("Scrap for Images", value=st.session_state.scrap_images, key="scrap_images_checkbox")

# Update session state with inputs and checkbox states
st.session_state.asset_id = asset_id
st.session_state.instrument = instrument
st.session_state.uploaded_files = uploaded_files
st.session_state.edit_uploaded = edit_uploaded
st.session_state.scrap_images = scrap_images

# Load API keys securely
API_KEYS = list(st.secrets.get("api_keys", {}).values())

if st.button("Process Images", key="process_button"):
    if not API_KEYS:
        st.error("API keys not configured. Add them in app settings.")
    elif not (edit_uploaded or scrap_images):
        st.error("Please select at least one option: Edit Uploaded Images or Scrap for Images.")
    else:
        with st.spinner("Fetching and processing images..."):
            try:
                output_pil_images, file_names, source_tags = process_instrument(asset_id, instrument, uploaded_files, edit_uploaded, scrap_images)
                if output_pil_images:
                    # Store processed data in session state
                    st.session_state.processed_data = {
                        'output_pil_images': output_pil_images,
                        'file_names': file_names,
                        'source_tags': source_tags
                    }
                    st.success(f"Processed {len(output_pil_images)} images!")
                else:
                    st.error("No images processed. Check logs or try different input.")
                    st.session_state.processed_data = None
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logging.error(f"Processing error: {e}")
                st.session_state.processed_data = None

# Display images if they exist in session state
if st.session_state.processed_data:
    output_pil_images = st.session_state.processed_data['output_pil_images']
    file_names = st.session_state.processed_data['file_names']
    source_tags = st.session_state.processed_data['source_tags']
    
    # Split into Uploaded and Scraped sections
    uploaded_images = [img for i, img in enumerate(output_pil_images) if source_tags[i] == "Uploaded"]
    scraped_images = [img for i, img in enumerate(output_pil_images) if source_tags[i] == "Scraped"]
    uploaded_names = [name for i, name in enumerate(file_names) if source_tags[i] == "Uploaded"]
    scraped_names = [name for i, name in enumerate(file_names) if source_tags[i] == "Scraped"]

    if uploaded_images:
        st.subheader("Uploaded Images")
        for i, (pil_img, file_name) in enumerate(zip(uploaded_images, uploaded_names)):
            view_index = i // NUM_IMAGES_PER_VIEW
            view = VIEWS[view_index] if view_index < len(VIEWS) else "extra"
            st.image(pil_img, caption=f"{view.capitalize()} View {i % NUM_IMAGES_PER_VIEW + 1}")
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            st.download_button(
                label=f"Download {file_name}",
                data=img_bytes,
                file_name=file_name,
                mime="image/jpeg",
                key=f"download_uploaded_{file_name}_{i}"  # Unique key with file_name
            )

    if scraped_images:
        st.subheader("Scraped Images")
        for i, (pil_img, file_name) in enumerate(zip(scraped_images, scraped_names)):
            view_index = i // NUM_IMAGES_PER_VIEW
            view = VIEWS[view_index] if view_index < len(VIEWS) else "extra"
            st.image(pil_img, caption=f"{view.capitalize()} View {i % NUM_IMAGES_PER_VIEW + 1}")
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            st.download_button(
                label=f"Download {file_name}",
                data=img_bytes,
                file_name=file_name,
                mime="image/jpeg",
                key=f"download_scraped_{file_name}_{i}"  # Unique key with file_name
            )

    # ZIP download - create from PIL images in memory
    if st.button("Download All Images as ZIP", key="download_zip_button"):
        try:
            zip_buffer = create_zip_from_images(output_pil_images, file_names, asset_id)
            st.download_button(
                label="Download All Images as ZIP",
                data=zip_buffer,
                file_name=f"{asset_id}_images.zip",
                mime="application/zip",
                key="zip_download"
            )
        except Exception as e:
            st.error(f"Failed to create ZIP: {str(e)}")
            logging.error(f"ZIP creation error: {e}")
