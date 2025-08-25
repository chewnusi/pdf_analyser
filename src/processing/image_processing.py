import numpy as np
import streamlit as st
import pytesseract
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from ..models.model_loader import load_paligemma_model_and_processor
import torch

def find_main_content_by_text_clustering(image: Image.Image, padding=50):
    """
    Finds the main content area by clustering OCR text boxes and selecting the largest cluster.
    """
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(ocr_data['level'])
        if n_boxes == 0: return None, None

        boxes = []
        centers = []
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 20 and ocr_data['text'][i].strip() != "":
                x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
                boxes.append((x, y, x + w, y + h))
                centers.append((x + w / 2, y + h / 2))

        if not centers: return None, None

        eps_distance = max(image.width, image.height) * 0.1
        clustering = DBSCAN(eps=eps_distance, min_samples=2).fit(centers)
        
        labels = clustering.labels_
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) == 0:
            return None, None

        largest_cluster_label = unique_labels[counts.argmax()]
        
        main_cluster_boxes = [box for i, box in enumerate(boxes) if labels[i] == largest_cluster_label]
        
        cluster_viz = image.copy()
        draw = ImageDraw.Draw(cluster_viz)
        colors = ["green", "blue", "purple", "orange"]
        for i, box in enumerate(boxes):
            if labels[i] == largest_cluster_label:
                draw.rectangle(box, outline="red", width=3)
            elif labels[i] != -1:
                color = colors[labels[i] % len(colors)]
                draw.rectangle(box, outline=color, width=1)

        min_x = min([box[0] for box in main_cluster_boxes])
        min_y = min([box[1] for box in main_cluster_boxes])
        max_x = max([box[2] for box in main_cluster_boxes])
        max_y = max([box[3] for box in main_cluster_boxes])

        box_with_padding = (
            max(0, min_x - 50),            
            max(0, min_y - 250),           
            min(image.width, max_x + 350),  
            min(image.height, max_y + 400)  
        )
        
        return box_with_padding, cluster_viz

    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or not in your PATH. Please install Tesseract-OCR.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during OCR/Clustering: {e}")
        return None, None

def find_vertical_splits(image: Image.Image, threshold_ratio=0.99, min_split_width=250, top_n=5):
    """
    Finds up to N of the most significant vertical whitespace splits.
    """
    gs_array = np.array(image.convert('L'))
    
    vertical_projection = np.sum(gs_array, axis=0) / 255
    white_threshold = np.max(vertical_projection) * threshold_ratio
    is_white = vertical_projection > white_threshold
    
    gaps = []
    in_gutter = False
    for i, is_col_white in enumerate(is_white):
        if is_col_white and not in_gutter:
            in_gutter = True
            gutter_start = i
        elif not is_col_white and in_gutter:
            in_gutter = False
            width = i - gutter_start
            if width > min_split_width:
                score = width * np.mean(vertical_projection[gutter_start:i])
                gaps.append({'start': gutter_start, 'end': i, 'width': width, 'score': score})
    
    if not gaps:
        return []
        
    sorted_gaps = sorted(gaps, key=lambda x: x['score'], reverse=True)
    return [gap['start'] + gap['width'] // 2 for gap in sorted_gaps[:top_n]]

def generate_table_name(cropped_table_image: Image.Image, model, processor) -> str:
    """
    Generates a descriptive name for a table image using PaliGemma.
    """
    try:
        prompt = "Generate a title based on table header."
        inputs = processor(
            text=prompt, 
            images=cropped_table_image, 
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generation_output = model.generate(**inputs, max_new_tokens=48)
        
        result = processor.decode(generation_output[0], skip_special_tokens=True)
        
        final_name = result.replace(prompt, "").strip()

        return final_name if final_name else "Table (PaliGemma)"

    except Exception as e:
        st.error(f"PaliGemma Error: {e}")
        st.warning("PaliGemma requires a GPU and significant memory. Ensure your environment is set up correctly.")
        return "Table (Naming Failed)"
