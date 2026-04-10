import streamlit as st
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import time
from model import Colorizer # Ensure model.py is in the same folder

# --- 1. UI CONFIG ---
st.set_page_config(page_title="NeuralColorizer", page_icon="🎨")

# Custom CSS to keep the layout tight
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #238636; color: white; }
    .stImage > img { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Colorizer().to(device)
    # weights_only=True removes the pickle warning
    model.load_state_dict(torch.load('landscape_model.pth', map_location=device, weights_only=True))
    model.eval()
    return model, device

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    res_slider = st.select_slider("Inference Quality", options=[128, 256], value=128)
    st.info("RTX GPU Active" if torch.cuda.is_available() else "Running on CPU")
    st.caption("M.Tech Data Science Project")

# --- 4. MAIN INTERFACE ---
st.title("🎨 Neural Image Colorizer")

uploaded_file = st.file_uploader("Upload a B&W Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Load original image and store its size
    original_img = Image.open(uploaded_file).convert('RGB')
    orig_w, orig_h = original_img.size
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Input")
        st.image(original_img, use_column_width=True)
    
    if st.button("Generate Color"):
        model, device = load_model()
        start = time.time()
        
        with st.spinner("Colorizing..."):
            # 2. Resize to processing resolution
            img_resized = original_img.resize((res_slider, res_slider))
            img_np = np.array(img_resized) / 255.0
            
            # 3. LAB conversion
            lab = rgb2lab(img_np)
            L = lab[:, :, 0:1] / 100.0
            L_tensor = torch.from_numpy(L).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # 4. Inference
            with torch.no_grad():
                pred_ab = model(L_tensor)
            
            # 5. Reconstruction
            pred_ab = pred_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128.0
            combined = np.concatenate((L * 100.0, pred_ab), axis=2)
            result_rgb = (lab2rgb(combined) * 255).astype(np.uint8)
            
            # 6. CRITICAL: Resize back to original pixel dimensions
            # Image.LANCZOS ensures high quality for your M.Tech demo
            final_output = Image.fromarray(result_rgb).resize((orig_w, orig_h), Image.LANCZOS)
            
        runtime = time.time() - start
        
        with col2:
            st.markdown("#### Output")
            st.image(final_output, use_column_width=True)
        
        st.metric("Inference Time", f"{runtime:.3f} seconds")