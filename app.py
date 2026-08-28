import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
from gar import GradientAttentionRollout

# Page Configuration
st.set_page_config(
    page_title="Deepfake GAR Explainer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Deepfake Gradient Attention Rollout (GAR)")
st.markdown("""
This dashboard extracts token-level attention from a Vision Transformer (ViT) 
and projects gradient-weighted heatmaps highlighting regions driving deepfake predictions.
""")

# Explicit attention forward function so autograd tracks gradients
def explicit_attention_forward(self, x, *args, **kwargs):
    B, N, C = x.shape
    num_heads = self.num_heads
    head_dim = C // num_heads
    
    # Project Q, K, V
    qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Scaled Dot-Product Attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Attention Output
    out = torch.matmul(attn_weights, v)
    out = out.permute(0, 2, 1, 3).reshape(B, N, C)
    out = self.out_proj(out)
    
    return out, attn_weights

# Cache model loading so Streamlit doesn't reload PyTorch on every interaction
@st.cache_resource
def load_model():
    model = torchvision.models.vit_b_16(weights='DEFAULT')
    target_layer = model.encoder.layers[-1].self_attention
    target_layer.forward = explicit_attention_forward.__get__(target_layer)
    return model, target_layer

model, target_layer = load_model()
gar_extractor = GradientAttentionRollout(model, target_layer)

# Image preprocessing transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sidebar options
st.sidebar.header("Pipeline Controls")
alpha = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
colormap = st.sidebar.selectbox("Colormap", ["jet", "viridis", "inferno", "magma", "plasma"])

# Upload Section
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load image
    raw_img = Image.open(uploaded_file).convert("RGB")
    display_img = raw_img.resize((224, 224))
    
    input_tensor = transform(raw_img).unsqueeze(0)
    input_tensor.requires_grad = True

    # 2. Run GAR Pipeline
    with st.spinner("Extracting Attention Maps & Gradients..."):
        heatmap_numpy = gar_extractor.generate_mask(input_tensor)

    # 3. Create Heatmap Overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_img)
    ax.imshow(heatmap_numpy, cmap=colormap, alpha=alpha)
    ax.axis("off")
    plt.tight_layout()

    # 4. Display Results in Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(display_img, use_container_width=True)

    with col2:
        st.subheader("Raw GAR Array")
        st.image(heatmap_numpy, clamp=True, use_container_width=True)
        st.caption("2D Normalized Attention Array [0, 1]")

    with col3:
        st.subheader("GAR Visual Overlay")
        st.pyplot(fig, use_container_width=True)

    # Metrics section
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Heatmap Resolution", value=f"{heatmap_numpy.shape[0]}x{heatmap_numpy.shape[1]}")
    m2.metric(label="Peak Attention Score", value=f"{heatmap_numpy.max():.4f}")
    m3.metric(label="Min Attention Score", value=f"{heatmap_numpy.min():.4f}")

    # Developer handoff section for Krishna
    with st.expander("🛠️ Developer Data Handoff (For Krishna's MediaPipe Mesh)"):
        st.write("This raw 2D NumPy array is ready to be multiplied or aligned with MediaPipe face landmark coordinates:")
        st.code(f"Array Shape: {heatmap_numpy.shape}\nData Type: {heatmap_numpy.dtype}\nSample values:\n{heatmap_numpy[:3, :3]}")

else:
    st.info("Upload an image above to run the GAR pipeline and generate heatmaps.")