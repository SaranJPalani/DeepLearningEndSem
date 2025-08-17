# Complete Damage Segmentation Inference Script
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Define the UNetPP model architecture (COPY THIS PART)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class UNetPP(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, filters=(32,64,128,256,512), deep_supervision=False, dropout_rates=None):
        super().__init__()
        self.deep_supervision = deep_supervision
        f = filters
        
        # Define dropout rates for different layers
        if dropout_rates is None:
            dropout_rates = {
                'encoder': [0.0, 0.0, 0.0, 0.1, 0.15],
                'decoder': [0.3, 0.25, 0.2, 0.15],
                'skip': 0.2,
                'final': 0.2
            }
        
        # Encoder base layers
        self.x_0_0 = ConvBlock(in_ch, f[0], dropout_rates['encoder'][0])
        self.x_1_0 = ConvBlock(f[0], f[1], dropout_rates['encoder'][1])
        self.x_2_0 = ConvBlock(f[1], f[2], dropout_rates['encoder'][2])
        self.x_3_0 = ConvBlock(f[2], f[3], dropout_rates['encoder'][3])
        self.x_4_0 = ConvBlock(f[3], f[4], dropout_rates['encoder'][4])
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Nested decoder blocks
        self.x_0_1 = ConvBlock(f[0]+f[1], f[0], dropout_rates['skip'])
        self.x_1_1 = ConvBlock(f[1]+f[2], f[1], dropout_rates['skip'])
        self.x_2_1 = ConvBlock(f[2]+f[3], f[2], dropout_rates['skip'])
        self.x_3_1 = ConvBlock(f[3]+f[4], f[3], dropout_rates['decoder'][0])

        self.x_0_2 = ConvBlock(f[0]*2+f[1], f[0], dropout_rates['skip'])
        self.x_1_2 = ConvBlock(f[1]*2+f[2], f[1], dropout_rates['skip'])
        self.x_2_2 = ConvBlock(f[2]*2+f[3], f[2], dropout_rates['decoder'][1])

        self.x_0_3 = ConvBlock(f[0]*3+f[1], f[0], dropout_rates['skip'])
        self.x_1_3 = ConvBlock(f[1]*3+f[2], f[1], dropout_rates['decoder'][2])

        self.x_0_4 = ConvBlock(f[0]*4+f[1], f[0], dropout_rates['decoder'][3])

        # Final layers
        self.final_dropout = nn.Dropout2d(dropout_rates['final'])
        self.final = nn.Conv2d(f[0], out_ch, 1)
    
    def forward(self, x):
        # Encoder path
        x_0_0 = self.x_0_0(x)
        x_1_0 = self.x_1_0(self.pool(x_0_0))
        x_0_1 = self.x_0_1(torch.cat([x_0_0, self.up(x_1_0)], dim=1))

        x_2_0 = self.x_2_0(self.pool(x_1_0))
        x_1_1 = self.x_1_1(torch.cat([x_1_0, self.up(x_2_0)], dim=1))
        x_0_2 = self.x_0_2(torch.cat([x_0_0, x_0_1, self.up(x_1_1)], dim=1))

        x_3_0 = self.x_3_0(self.pool(x_2_0))
        x_2_1 = self.x_2_1(torch.cat([x_2_0, self.up(x_3_0)], dim=1))
        x_1_2 = self.x_1_2(torch.cat([x_1_0, x_1_1, self.up(x_2_1)], dim=1))
        x_0_3 = self.x_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up(x_1_2)], dim=1))

        # Bottleneck
        x_4_0 = self.x_4_0(self.pool(x_3_0))
        
        # Decoder path
        x_3_1 = self.x_3_1(torch.cat([x_3_0, self.up(x_4_0)], dim=1))
        x_2_2 = self.x_2_2(torch.cat([x_2_0, x_2_1, self.up(x_3_1)], dim=1))
        x_1_3 = self.x_1_3(torch.cat([x_1_0, x_1_1, x_1_2, self.up(x_2_2)], dim=1))
        x_0_4 = self.x_0_4(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, self.up(x_1_3)], dim=1))

        # Final prediction
        out = self.final_dropout(x_0_4)
        out = self.final(out)
        return out

# Load your trained model
very_aggressive_dropout = {
    'encoder': [0.0, 0.15, 0.25, 0.4, 0.6],
    'decoder': [0.6, 0.5, 0.4, 0.3],
    'skip': 0.4,
    'final': 0.4
}

# Initialize model
model = UNetPP(in_ch=6, out_ch=1, filters=(32,64,128,256,512), 
               dropout_rates=very_aggressive_dropout).to(DEVICE)

# Load trained weights (make sure 'best_unetpp.pth' is in the same directory)
try:
    model.load_state_dict(torch.load('best_unetpp.pth', map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: 'best_unetpp.pth' not found! Make sure the model file is in the same directory.")
    exit()

# Image preprocessing (same as training)
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def segment_damage(pre_image_path, post_image_path, threshold=0.3, save_output=True, output_dir='damage_results'):
    """
    Segment earthquake damage between pre and post disaster images
    """
    
    # Check if files exist
    if not os.path.exists(pre_image_path):
        print(f"ERROR: Pre-disaster image not found: {pre_image_path}")
        return None, None
    if not os.path.exists(post_image_path):
        print(f"ERROR: Post-disaster image not found: {post_image_path}")
        return None, None
    
    # Load and preprocess images
    try:
        pre_img = Image.open(pre_image_path).convert('RGB')
        post_img = Image.open(post_image_path).convert('RGB')
        print(f"Loaded images: {pre_img.size} -> {post_img.size}")
    except Exception as e:
        print(f"ERROR loading images: {e}")
        return None, None
    
    # Transform images
    pre_tensor = transform_img(pre_img).unsqueeze(0)  # Add batch dimension
    post_tensor = transform_img(post_img).unsqueeze(0)
    
    # Concatenate pre and post (6 channels total)
    input_tensor = torch.cat([pre_tensor, post_tensor], dim=1).to(DEVICE)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        confidence_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # Remove batch dim
        damage_mask = (confidence_map >= threshold).astype(np.uint8)
    
    print(f"Confidence range: {confidence_map.min():.3f} - {confidence_map.max():.3f}")
    
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axes[0,0].imshow(pre_img)
        axes[0,0].set_title('Pre-Disaster')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(post_img)
        axes[0,1].set_title('Post-Disaster')
        axes[0,1].axis('off')
        
        # Confidence map (heatmap)
        im1 = axes[0,2].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
        axes[0,2].set_title(f'Damage Confidence')
        axes[0,2].axis('off')
        plt.colorbar(im1, ax=axes[0,2])
        
        # Binary damage mask
        axes[1,0].imshow(damage_mask, cmap='Reds')
        axes[1,0].set_title(f'Damage Mask (thresh={threshold})')
        axes[1,0].axis('off')
        
        # Overlay on post-disaster image
        post_array = np.array(post_img.resize((256, 256)))
        overlay = post_array.copy()
        overlay[damage_mask == 1] = [255, 0, 0]  # Red for damage
        axes[1,1].imshow(overlay)
        axes[1,1].set_title('Damage Overlay')
        axes[1,1].axis('off')
        
        # Statistics
        axes[1,2].axis('off')
        total_pixels = damage_mask.shape[0] * damage_mask.shape[1]
        damage_pixels = np.sum(damage_mask)
        damage_percentage = (damage_pixels / total_pixels) * 100
        
        stats_text = f"""
        DAMAGE ANALYSIS
        ===============
        Total Pixels: {total_pixels:,}
        Damage Pixels: {damage_pixels:,}
        Damage Area: {damage_percentage:.2f}%
        
        Confidence Stats:
        Max: {confidence_map.max():.3f}
        Mean: {confidence_map.mean():.3f}
        Min: {confidence_map.min():.3f}
        """
        axes[1,2].text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                      verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Save results
        base_name = os.path.splitext(os.path.basename(post_image_path))[0]
        plt.savefig(f'{output_dir}/{base_name}_damage_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save individual outputs
        Image.fromarray((damage_mask * 255).astype(np.uint8)).save(f'{output_dir}/{base_name}_damage_mask.png')
        Image.fromarray((confidence_map * 255).astype(np.uint8)).save(f'{output_dir}/{base_name}_confidence.png')
        
        print(f"Results saved in '{output_dir}/' directory")
        print(f"Damage detected in {damage_percentage:.2f}% of the area")
    
    return damage_mask, confidence_map

# Example usage
if __name__ == "__main__":
    # Test with sample images (CHANGE THESE PATHS)
    pre_path = "pre/train(2).png"
    post_path = "post/train(2).png" 
    
    print("Starting damage segmentation...")
    
    # Run segmentation
    damage_mask, confidence_map = segment_damage(
        pre_image_path=pre_path,
        post_image_path=post_path,
        threshold=0.3,
        save_output=True
    )
    
    if damage_mask is not None:
        print("✅ Segmentation completed successfully!")
    else:
        print("❌ Segmentation failed!")