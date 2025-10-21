import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
# Import your custom preprocessing functions (hair removal, CLAHE)
# from your_notebook_utils import remove_hair, apply_clahe 

# --- 1. Define Your Model Architecture ---
# This MUST be the *exact* same architecture as in your notebook
# Here, I'm assuming you used 'timm' to create your EfficientNet-B3
class SimpleEfficientNet(nn.Module):
    """
    Simple EfficientNet-B3 baseline classifier.
    
    This simple architecture OUTPERFORMED complex multi-scale + attention models:
    - 97.98% recall (vs 93.66% for complex model)
    - 64.4% coverage (vs 56.3% for complex model)
    - Only 227 uncertain malignant cases (vs 382 for complex model)
    
    Architecture:
    - EfficientNet-B3 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B3
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            base_model = efficientnet_b3(weights=weights)
        else:
            base_model = efficientnet_b3(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # EfficientNet-B3 outputs 1536 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# --- 2. Load Your Trained Model ---
model_path = 'best_melanoma_improved.pth'
model = SimpleEfficientNet(num_classes=2, pretrained=False)
# Load checkpoint (contains model_state_dict, val_auc, val_acc, epoch)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode

# --- 3. Define Your Preprocessing Pipeline ---
# This MUST match the validation/test transforms from your notebook
# NOTE: Gradio provides a PIL Image, so no need for ToPILImage()
preprocess_transform = transforms.Compose([
    transforms.Resize((300, 300)), # You used 300x300 in your notebook
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 4. Create the Prediction Function ---
def predict(input_image):
    """
    Takes a PIL Image as input, preprocesses it, and returns a 
    dictionary of labels and their confidence scores.
    """
    # Apply your custom preprocessing (e.g., hair removal)
    # image_no_hair = remove_hair(input_image) 
    # image_clahe = apply_clahe(image_no_hair)
    # For this example, I'll just use the input image
    
    # Apply PyTorch transforms
    input_tensor = preprocess_transform(input_image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    # Run prediction
    with torch.no_grad():
        output = model(input_batch)
        # Apply softmax to get probabilities for 2-class output
        probabilities = torch.softmax(output, dim=1)[0]
        benign_prob = probabilities[0].item()
        malignant_prob = probabilities[1].item()

    # Format the output for Gradio's "label" component
    return {'Benign': benign_prob, 'Malignant': malignant_prob}

# --- 5. Launch the Gradio Interface ---
demo = gr.Interface(
    fn=predict,                     # The function to wrap
    inputs=gr.Image(type="pil"),    # Input is a PIL image
    outputs=gr.Label(num_top_classes=2), # Output is a "Label" component
    title="Melanoma Detection Demo",
    description="Upload a dermoscopy image to classify it as benign or malignant. \
                 (Proof-of-concept only. Not for medical use.)"
)

if __name__ == "__main__":
    demo.launch()