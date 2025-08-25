import torch
import streamlit as st
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection, 
    AutoModelForCausalLM, 
    AutoProcessor,
    PaliGemmaForConditionalGeneration
)
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_florence_model_and_processor():
    model_id = "ucsahin/Florence-2-large-TableDetection"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

@st.cache_resource
def load_yolov8_model():
    model_repo = "foduucom/table-detection-and-extraction"
    model_filename = "best.pt"
    model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_table_transformer_model_and_processor():
    processor = AutoImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
    model = AutoModelForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")
    return processor, model

@st.cache_resource
def load_paligemma_model_and_processor():
    """Loads the PaliGemma model and processor, caching it in Streamlit."""
    model_id = "google/paligemma-3b-mix-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency
        device_map="auto",  # Automatically use GPU if available
        revision="bfloat16",
        # torch_dtype=torch.float32,  # Use full precision
        # device_map="auto"  # Automatically use GPU if available
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
