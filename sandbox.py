import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import math
from PIL import Image
from gar import GradientAttentionRollout

def explicit_attention_forward(self, x, *args, **kwargs):
    """
    Explicitly computes Q, K, V attention math so PyTorch's autograd graph 
    tracks gradients through the attention matrix during backward passes.
    """
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

def run_sandbox_test():
    print("Loading PyTorch ViT model...")
    model = torchvision.models.vit_b_16(weights='DEFAULT')

    # Target the last self-attention layer
    target_layer = model.encoder.layers[-1].self_attention

    # Bind explicit attention forward pass
    target_layer.forward = explicit_attention_forward.__get__(target_layer)

    # Initialize GAR Extractor
    gar_extractor = GradientAttentionRollout(model, target_layer)

    # Preprocess Image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open("test_image.jpg").convert("RGB")
        input_tensor = transform(raw_img).unsqueeze(0)
        print("Loaded 'test_image.jpg' successfully.")
    except FileNotFoundError:
        print("Notice: 'test_image.jpg' not found. Using a random tensor for testing.")
        input_tensor = torch.rand(1, 3, 224, 224)

    input_tensor.requires_grad = True

    print("Running GAR Pipeline...")
    mask = gar_extractor.generate_mask(input_tensor)

    print("\n--- TEST RESULTS ---")
    print(f"Mask Data Type : {type(mask)}")
    print(f"Mask Shape     : {mask.shape}")
    print(f"Value Range    : Min = {mask.min():.4f}, Max = {mask.max():.4f}")
    print("--------------------")
    print("Success! Your GAR array is ready for Streamlit and MediaPipe.")

    gar_extractor.remove_hooks()

if __name__ == "__main__":
    run_sandbox_test()