# Low-Cost Deep Learning System for Automated Melanoma Detection
# Copyright (C) 2025 Nicolò Calandra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Import model architecture
from models_code.EfficientNets import SimpleEfficientNet, SimpleEfficientNetB4
from models_code.ConvNeXt import SimpleConvNeXtBase

# Import medical preprocessing
from medical_preprocessing_final import InpaintingMaskFiller, HairRemoval, ContrastEnhancement

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = 'best_melanoma_single.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
# model = SimpleEfficientNetB4(num_classes=2, pretrained=False)
model = SimpleConvNeXtBase(num_classes=2, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Medical preprocessing pipeline (applied to raw images)
medical_preprocessing = transforms.Compose([
    InpaintingMaskFiller(threshold=50, inpaint_radius=15, min_black_ratio=0.001),
    HairRemoval(kernel_size=17),
    ContrastEnhancement(clip_limit=2.0)
])

# Model input preprocessing (applied after medical preprocessing)
model_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# PREDICTION & UI HELPER FUNCTIONS
# ============================================================

def get_risk_level(malignant_prob):
    """Categorize risk level and return style info."""
    if malignant_prob >= 0.70:
        return "High Risk", "#ef4444", "recommendation-box-high"
    elif malignant_prob >= 0.30:
        return "Moderate Risk", "#f59e0b", "recommendation-box-moderate"
    else:
        return "Low Risk", "#10b981", "recommendation-box-low"

def create_probability_bars_html(benign_prob, malignant_prob):
    """Create HTML for the benign/malignant probability bars."""
    benign_pct = benign_prob * 100
    malignant_pct = malignant_prob * 100
    
    return f"""
    <div class="prob-container">
        <div class="prob-bar-label">
            <span>Benign</span>
            <span class="prob-bar-percent">{benign_pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: {benign_pct}%; background-color: #10b981;"></div>
        </div>
    </div>
    <div class="prob-container">
        <div class="prob-bar-label">
            <span>Malignant</span>
            <span class="prob-bar-percent">{malignant_pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: {malignant_pct}%; background-color: #ef4444;"></div>
        </div>
    </div>
    """

def predict(input_image, confidence_threshold):
    """
    Analyze dermoscopy image and return styled HTML components for the UI.
    """
    if input_image is None:
        # Return empty strings for all HTML outputs
        return None, (
            "", "", "", "", None, None
        )
    
    # Apply medical preprocessing
    try:
        preprocessed_image = medical_preprocessing(input_image)
    except Exception as e:
        print(f"Warning: Preprocessing failed ({e}), using original image")
        preprocessed_image = input_image
    
    # Apply model-specific transforms and predict
    input_tensor = model_transform(preprocessed_image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.softmax(output, dim=1)[0].cpu()
        benign_prob = probabilities[0].item()
        malignant_prob = probabilities[1].item()
    
    # Determine prediction and risk
    max_prob = max(benign_prob, malignant_prob)
    prediction = "MALIGNANT" if malignant_prob > benign_prob else "BENIGN"
    risk_level, risk_color, rec_box_class = get_risk_level(malignant_prob)
    
    # Check confidence
    is_confident = max_prob >= (confidence_threshold / 100)
    
    # --- 1. Create Result Box HTML ---
    result_box_html = f"""
    <div class="result-box" style="border-color: {risk_color};">
        <div class="result-box-prediction" style="color: {risk_color};">{prediction}</div>
        <div class="result-box-risk">Risk Level: <strong>{risk_level}</strong></div>
    </div>
    """
    
    # --- 2. Create Probability Bars HTML ---
    prob_bars_html = create_probability_bars_html(benign_prob, malignant_prob)
    
    # --- 3. Create Status Box HTML ---
    if is_confident:
        status_icon = "✓"
        status_color = "#10b981"
        status_text = f"<strong>Confident Prediction</strong> ({max_prob*100:.1f}%)"
        status_box_class = "status-box-confident"
    else:
        status_icon = "⚠️"
        status_color = "#f59e0b"
        status_text = f"<strong>Uncertain</strong> - Confidence below {confidence_threshold}% ({max_prob*100:.1f}%)"
        status_box_class = "status-box-uncertain"
        
    status_box_html = f"""
    <div class="status-box {status_box_class}">
        <span style="color: {status_color}; font-size: 1.5em; margin-right: 10px;">{status_icon}</span>
        <span>{status_text}</span>
    </div>
    """
    
    # --- 4. Create Recommendation HTML ---
    if malignant_prob >= 0.70:
        recommendation_icon = "⚠️"
        recommendation_text = "<strong>HIGH PRIORITY:</strong> Immediate dermatologist consultation is strongly recommended."
    elif malignant_prob >= 0.30:
        recommendation_icon = "🔍"
        recommendation_text = "<strong>MONITOR:</strong> Schedule a follow-up examination with a dermatologist within 1-2 months."
    else:
        recommendation_icon = "✓"
        recommendation_text = "<strong>LOW CONCERN:</strong> Routine skin monitoring is recommended."

    recommendation_box_html = f"""
    <div class="recommendation-box {rec_box_class}">
        <span class="recommendation-icon">{recommendation_icon}</span>
        <p>{recommendation_text}</p>
    </div>
    """
    
    # --- 5. Create Preprocessing Info ---
    # (This is now static text in the "Processing Details" tab)
    
    # Return all components
    return (
        result_box_html, 
        prob_bars_html, 
        status_box_html, 
        recommendation_box_html, 
        input_image, 
        preprocessed_image
    )

# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """
/* --- Main Layout & Theme --- */
#main-container {
    max-width: 1400px;
    margin: auto;
    padding-top: 1.5rem;
}

/* Apply dark card styling to Groups */
.gradio-group {
    border: 1px solid rgba(75, 85, 99, 0.3);
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    padding: 24px !important;
    margin-top: 20px;
    background: rgba(31, 41, 55, 0.6);
    backdrop-filter: blur(10px);
}

.gradio-tabs {
    border: none !important;
    box-shadow: none !important;
}

/* --- Header --- */
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0c4a6e 100%);
    padding: 3rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.header-box h1 {
    font-size: 3.2em;
    margin: 0;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.header-box p {
    font-size: 1.35em;
    margin: 12px 0 0 0;
    color: #cbd5e1;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* --- Warning & Info Boxes --- */
.warning-box {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(245, 158, 11, 0.15) 100%);
    border: 1px solid rgba(251, 191, 36, 0.4);
    border-left: 4px solid #fbbf24;
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    color: #fcd34d !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(251, 191, 36, 0.1);
}

.warning-box strong {
    color: #fde047;
    font-size: 1.1em;
}

.info-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-left: 4px solid #3b82f6;
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    color: #93c5fd !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
}

.info-box strong {
    color: #bfdbfe;
    font-size: 1.1em;
}

.info-box ul {
    color: #93c5fd !important;
    margin: 10px 0 0 20px;
}

/* --- Buttons --- */
.primary-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.5) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 16px 32px !important;
    font-size: 1.15em !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.primary-btn:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 32px rgba(59, 130, 246, 0.5) !important;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
}

/* --- Image Upload --- */
.image-upload {
    border: 3px dashed rgba(100, 116, 139, 0.5);
    border-radius: 16px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(15, 23, 42, 0.4);
}

.image-upload:hover {
    border-color: #3b82f6;
    background: rgba(30, 41, 59, 0.6);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
}

/* --- Result Components (Right Column) --- */
.result-box {
    border: 2px solid;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(10px);
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.result-box-prediction {
    font-size: 2.8em;
    font-weight: 900;
    margin-bottom: 0.5rem;
    letter-spacing: 0.02em;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.result-box-risk {
    font-size: 1.35em;
    color: #e2e8f0 !important;
    font-weight: 500;
}

.status-box {
    display: flex;
    align-items: center;
    padding: 1.25rem;
    border-radius: 12px;
    font-size: 1.1em;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.status-box-confident {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #6ee7b7 !important;
}

.status-box-uncertain {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%);
    border: 1px solid rgba(251, 191, 36, 0.4);
    color: #fcd34d !important;
}

.recommendation-box {
    display: flex;
    align-items: flex-start;
    padding: 1.5rem;
    border-radius: 12px;
    font-size: 1.1em;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.recommendation-icon {
    font-size: 2em;
    margin-right: 14px;
    line-height: 1;
}

.recommendation-box p {
    margin: 0;
    line-height: 1.7;
}

.recommendation-box-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
    border: 1px solid rgba(248, 113, 113, 0.4);
    color: #fca5a5 !important;
}

.recommendation-box-moderate {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%);
    border: 1px solid rgba(251, 191, 36, 0.4);
    color: #fcd34d !important;
}

.recommendation-box-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.2) 100%);
    border: 1px solid rgba(52, 211, 153, 0.4);
    color: #6ee7b7 !important;
}

/* --- Probability Bars --- */
.prob-container {
    margin-bottom: 1.25rem;
}

.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 1em;
    font-weight: 600;
    color: #e2e8f0 !important;
    margin-bottom: 0.5rem;
}

.prob-bar-percent {
    font-weight: 700;
    color: #cbd5e1 !important;
}

.prob-bar-bg {
    width: 100%;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(71, 85, 105, 0.3);
    border-radius: 10px;
    height: 24px;
    overflow: hidden;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
}

.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
}

/* --- Footer --- */
.footer-text {
    text-align: center;
    color: #94a3b8;
    font-size: 1em;
    margin-top: 40px;
    padding: 24px;
    border-top: 1px solid rgba(71, 85, 105, 0.3);
}

.footer-text p {
    color: #94a3b8;
}

/* --- Global Text Improvements --- */
.gradio-markdown h3 {
    color: #f1f5f9 !important;
    font-weight: 700;
    margin-top: 0.5rem;
}

.gradio-markdown h4 {
    color: #cbd5e1 !important;
    font-weight: 600;
}
"""

# ============================================================
# GRADIO INTERFACE
# ============================================================

# Use a professional dark theme
theme = gr.themes.Soft(
    primary_hue="blue", 
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(to bottom, #0f172a, #1e293b)",
    body_background_fill_dark="linear-gradient(to bottom, #0f172a, #1e293b)",
    block_background_fill="*neutral_950",
    block_background_fill_dark="*neutral_950",
    block_border_width="1px",
    block_label_text_color="*neutral_200",
    block_label_text_color_dark="*neutral_200",
    block_title_text_color="*neutral_100",
    block_title_text_color_dark="*neutral_100",
    input_background_fill="*neutral_800",
    input_background_fill_dark="*neutral_800",
    button_primary_background_fill="linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
)

with gr.Blocks(css=custom_css, theme=theme) as demo:
    
    with gr.Column(elem_id="main-container"):
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1>🔬 Melanoma Detection AI</h1>
            <p>Advanced Deep Learning for Skin Lesion Analysis</p>
        </div>
        """)
        
        # Warning disclaimer
        gr.HTML("""
        <div class="warning-box">
            <strong>⚠️ Medical Disclaimer:</strong> This tool is for research and educational purposes only. 
            It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist 
            for any skin concerns.
        </div>
        """)
        
        with gr.Tabs():
            # --- TAB 1: AI Analyzer (Main App) ---
            with gr.TabItem("🔬 AI Analyzer"):
                with gr.Row(variant="panel"):
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload Dermoscopy Image")
                        image_input = gr.Image(
                            type="pil", 
                            label="Upload Image",
                            elem_classes="image-upload"
                        )
                        
                        confidence_slider = gr.Slider(
                            minimum=50,
                            maximum=95,
                            value=80,
                            step=5,
                            label="🎯 Confidence Threshold (%)",
                            info="Minimum confidence required for a definitive prediction"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze Image", 
                            variant="primary",
                            elem_classes="primary-btn"
                        )
                    
                    # Right column - Results
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Analysis Results")
                        
                        # --- New modular result components ---
                        result_box_html = gr.HTML(label="Prediction")
                        
                        gr.Markdown("#### Confidence Breakdown")
                        prob_bars_html = gr.HTML()
                        
                        gr.Markdown("#### Analysis Status")
                        status_box_html = gr.HTML()

                        gr.Markdown("#### Clinical Recommendation")
                        recommendation_box_html = gr.HTML()
            
            # --- TAB 2: Processing Details ---
            with gr.TabItem("⚙️ Processing Details"):
                with gr.Group():
                    gr.Markdown("### 🔬 Image Processing Pipeline")
                    gr.Markdown(
                        "The model automatically applies several preprocessing steps to "
                        "raw dermoscopy images to improve accuracy."
                    )
                    with gr.Row():
                        original_output = gr.Image(
                            label="Original Image",
                            type="pil"
                        )
                        preprocessed_output = gr.Image(
                            label="Preprocessed Image (Pre-analysis)",
                            type="pil"
                        )
                
                with gr.Group():
                    with gr.Accordion("📖 About This Model", open=True):
                        gr.Markdown("""
                        ### Model Architecture: ConvNeXt-Base
                        
                        This melanoma detection system uses a **ConvNeXt-Base** architecture 
                        with an automated medical image preprocessing pipeline.
                        
                        **Key Features:**
                        - **Pre-trained backbone** on ImageNet with medical domain fine-tuning
                        - **Automatic preprocessing** - hair removal, inpainting, contrast enhancement
                        - **High recall rate** (~98.8%) to minimize false negatives
                        - **Confidence-based filtering** to identify uncertain cases
                        - **Validated** on diverse dermoscopy datasets
                        
                        **Preprocessing Pipeline:**
                        1. **Black Corner Removal** - Inpainting to remove circular mask artifacts
                        2. **Hair Removal** - Morphological operations to remove hair artifacts
                        3. **Contrast Enhancement** - CLAHE for better lesion boundary definition
                        
                        **Training Details:**
                        - Dataset: ~10,000 annotated dermoscopy images
                        - Classes: Benign vs. Malignant lesions
                        - Augmentation: Rotation, color jittering, random crops
                        - Optimization: Progressive unfreezing with AdamW
                        """)

            # --- TAB 3: Examples & Info ---
            with gr.TabItem("📚 Examples & Info"):
                with gr.Group():
                    gr.HTML("""
                    <div class="info-box">
                        <strong>💡 Tips for Best Results:</strong>
                        <ul style="margin: 10px 0 0 20px;">
                            <li>Upload <strong>raw dermoscopy images</strong> - preprocessing is automatic!</li>
                            <li>Use high-quality images with good focus</li>
                            <li>The lesion should be centered in the frame</li>
                            <li>Hair and artifacts will be automatically removed</li>
                            <li>Black corners will be automatically filled</li>
                        </ul>
                    </div>
                    """)
                with gr.Group():
                    with gr.Accordion("🖼️ Try Example Images", open=True):
                        gr.Examples(
                            examples=[],  # Add your example image paths here
                            inputs=image_input,
                            label="Sample Dermoscopy Images"
                        )
        
        # Footer
        gr.HTML("""
        <div class="footer-text">
            <p><strong>Melanoma Detection AI</strong> | Powered by ConvNeXt-Base</p>
            <p>Research prototype - Not approved for clinical use</p>
        </div>
        """)
    
    # Connect the button to the prediction function
    analyze_btn.click(
        fn=predict,
        inputs=[image_input, confidence_slider],
        outputs=[
            result_box_html,
            prob_bars_html,
            status_box_html,
            recommendation_box_html,
            original_output,
            preprocessed_output
        ]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

