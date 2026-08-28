import torch
import torch.nn.functional as F
import numpy as np
import math

class GradientAttentionRollout:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        
        self.attention_maps = []
        self.attention_grads = []

        self.forward_hook = target_layer.register_forward_hook(self.save_attention)

    def save_attention(self, module, input, output):
        # Grabs attention weights tuple
        attn_weights = output[1] if isinstance(output, tuple) else output
        
        self.attention_maps.append(attn_weights.detach())

        # Tensor hook to record gradients during backward pass
        def _record_gradients(grad):
            self.attention_grads.append(grad.detach())
            
        attn_weights.register_hook(_record_gradients)

    def generate_mask(self, input_image, class_idx=None):
        self.attention_maps = []
        self.attention_grads = []
        self.model.zero_grad()

        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target_score = output[0, class_idx]
        target_score.backward()

        attn = self.attention_maps[-1]
        grads = self.attention_grads[-1]

        # Weight attention by gradients and average over heads
        weighted_attention = attn * grads
        weighted_attention = weighted_attention.mean(dim=1)
        weighted_attention = torch.clamp(weighted_attention, min=0)

        # CLS token row (skipping CLS self-attention at index 0)
        cls_attention = weighted_attention[0, 0, 1:] 

        # Spatial reshape
        num_patches = cls_attention.shape[0]
        grid_size = int(math.sqrt(num_patches))
        
        spatial_grid = cls_attention.reshape(grid_size, grid_size)
        spatial_grid = spatial_grid.unsqueeze(0).unsqueeze(0)
        
        # Upscale to match image dimensions
        upscaled_mask = F.interpolate(
            spatial_grid, 
            size=(input_image.shape[2], input_image.shape[3]), 
            mode='bilinear', 
            align_corners=False
        )

        heatmap_numpy = upscaled_mask.squeeze().cpu().numpy()
        heatmap_numpy = (heatmap_numpy - heatmap_numpy.min()) / (heatmap_numpy.max() - heatmap_numpy.min() + 1e-8)

        return heatmap_numpy

    def remove_hooks(self):
        self.forward_hook.remove()