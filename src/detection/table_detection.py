import numpy as np
import torch
import pdfplumber
import streamlit as st
import io

from ..models.model_loader import (
    load_florence_model_and_processor,
    load_yolov8_model,
    load_table_transformer_model_and_processor,
)

def apply_nms(boxes, scores, iou_threshold=0.2):
    """Applies Non-Maximum Suppression to filter overlapping boxes."""
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def detect_tables_florence(page_image):
    tables_info = []
    model, processor = load_florence_model_and_processor()
    def run_florence(image):
        inputs = processor(text="<OD>", images=image, return_tensors="pt")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].cuda(), pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024, num_beams=3)
        return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    generated_text = run_florence(page_image)
    parsed = processor.post_process_generation(generated_text, task="<OD>", image_size=page_image.size)
    
    if '<OD>' in parsed and 'bboxes' in parsed['<OD>']:
        for bbox in parsed['<OD>']['bboxes']:
            tables_info.append({"location": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}, "name": "Table (Florence)"})
    return tables_info

def detect_tables_yolov8(page_image):
    tables_info = []
    model = load_yolov8_model()
    model.overrides['conf'] = 0 
    model.overrides['iou'] = 0.9
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 1000
    results = model.predict(page_image)
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            x0, y0, x1, y1 = box.xyxy[0]
            tables_info.append({"location": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}, "name": f"{label.capitalize()} (YOLO)"})
    return tables_info

def detect_tables_table_transformer(page_image):
    tables_info = [] 
    processor, model = load_table_transformer_model_and_processor()
    inputs = processor(images=page_image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([page_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    boxes = np.array([box.tolist() for box in results["boxes"]])
    scores = np.array([s.item() for s in results["scores"]])
    if len(boxes) > 0:
        keep_indices = apply_nms(boxes, scores, iou_threshold=0.2)
        for i in keep_indices:
            box = boxes[i]
            x0, y0, x1, y1 = box
            tables_info.append({"location": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}})
    return tables_info

def is_contained(bbox_container, bbox_child):
    """Checks if bbox_child is completely inside bbox_container."""
    return (
        bbox_container["x0"] <= bbox_child["x0"] and
        bbox_container["y0"] <= bbox_child["y0"] and
        bbox_container["x1"] >= bbox_child["x1"] and
        bbox_container["y1"] >= bbox_child["y1"]
    )

def detect_tables_pdfplumber(pdf_bytes, page_image):
    """Detects tables using pdfplumber and translates coordinates to the image space."""
    all_tables_info = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return []
            page = pdf.pages[0]

            scale_x = page_image.width / float(page.width)
            scale_y = page_image.height / float(page.height)

            found_tables = page.find_tables()
                
            for i, table in enumerate(found_tables):
                bbox = table.bbox
                img_bbox = {
                    "x0": int(bbox[0] * scale_x), "y0": int(bbox[1] * scale_y),
                    "x1": int(bbox[2] * scale_x), "y1": int(bbox[3] * scale_y)
                }
                all_tables_info.append({"id": i, "location": img_bbox})

            if len(all_tables_info) < 2:
                return all_tables_info

            container_indices = set()
            tables_to_check = all_tables_info

            for i, potential_container in enumerate(tables_to_check):
                for j, potential_child in enumerate(tables_to_check):
                    if i == j: 
                        continue
                    
                    if potential_container["location"] == potential_child["location"]:
                        continue

                    if is_contained(potential_container["location"], potential_child["location"]):
                        container_indices.add(i)

            final_tables_info = []
            for i, table_info in enumerate(all_tables_info):
                if i not in container_indices:
                    table_info["name"] = "Table (pdfplumber)"
                    final_tables_info.append(table_info)
                
            return final_tables_info

    except Exception as e:
        st.error(f"An error occurred during pdfplumber processing: {e}")
    return []
