import streamlit as st
from PIL import ImageDraw
from src.models.model_loader import load_paligemma_model_and_processor
from src.utils.pdf_utils import validate_pdf, pdf_to_image, draw_bounding_boxes
from src.detection.table_detection import (
    detect_tables_florence,
    detect_tables_yolov8,
    detect_tables_table_transformer,
    detect_tables_pdfplumber
)
from src.processing.image_processing import (
    find_main_content_by_text_clustering,
    find_vertical_splits,
    generate_table_name
)

def run_detection(page_image, model_name, use_layout_analysis):
    detection_functions = {
        "YOLOv8": detect_tables_yolov8,
        "TableTransformer": detect_tables_table_transformer,
        "Florence-2": detect_tables_florence,
    }
    detect_func = detection_functions.get(model_name)
    if not detect_func:
        st.warning(f"Model '{model_choice}' is not implemented.")
        return [], []

    processing_steps = [("1. Original", page_image.copy())]
    all_tables = []
    
    image_to_process = page_image
    crop_offset_x, crop_offset_y = 0, 0
    
    if use_layout_analysis:
        content_box, cluster_viz = find_main_content_by_text_clustering(image_to_process)
        if content_box:
            processing_steps.append(("2. Text Clusters", cluster_viz))
            image_to_process = page_image.crop(content_box)
            crop_offset_x, crop_offset_y = content_box[0], content_box[1]
            processing_steps.append(("3. Final Crop", image_to_process.copy()))
        else:
            st.warning("Cropping failed, using full image.")

        v_splits = find_vertical_splits(image_to_process)
        splits_viz = image_to_process.copy()
        draw = ImageDraw.Draw(splits_viz)
        for x in v_splits:
            draw.line([(x, 0), (x, splits_viz.height)], fill="blue", width=5)
        processing_steps.append(("4. Detected Splits", splits_viz))

        x_boundaries = sorted([0] + v_splits + [image_to_process.width])
        
        for i in range(len(x_boundaries) - 1):
            x0, x1 = x_boundaries[i], x_boundaries[i+1]
            if (x1 - x0 < 50):
                continue

            segment = image_to_process.crop((x0, 0, x1, image_to_process.height))
            segment_tables = detect_func(segment)
            
            for table in segment_tables:
                bbox = table["location"]
                bbox["x0"] += crop_offset_x + x0
                bbox["y0"] += crop_offset_y  
                bbox["x1"] += crop_offset_x + x0
                bbox["y1"] += crop_offset_y
            all_tables.extend(segment_tables)
    else:
        all_tables = detect_func(page_image)

    return all_tables, processing_steps

def main():
    st.set_page_config(page_title="PDF Table Detector", layout="wide")
    st.title("📊 Visual Layout-Aware Table Detector")

    with st.sidebar:
        st.header("⚙️ Controls")
        
        processing_method = st.radio(
            "1. Choose Processing Method", 
            ('Computer Vision', 'Text Processor (pdfplumber)')
        )

        if processing_method == 'Computer Vision':
            model_choice = st.selectbox(
                "2. Choose a Detection Model",
                ("TableTransformer", "Florence-2", "YOLOv8")
            )
            st.header("✨ Pre-processing Pipeline")
            use_layout_analysis = st.checkbox(
                "Enable Layout Analysis",
                value=True,
                help="Automatically crop borders and split by wide white spaces before running detection."
            )
        else:
            model_choice = "pdfplumber"
            use_layout_analysis = False

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("1. Upload PDF")
        uploaded_file = st.file_uploader("Choose a single-page PDF", type=['pdf'])
        
        if uploaded_file:
            if st.button("🔍 Detect Tables", type="primary", use_container_width=True):
                pdf_bytes = uploaded_file.getvalue()
                is_valid, message = validate_pdf(uploaded_file)
                page_image = pdf_to_image(pdf_bytes)
                st.session_state.page_image = page_image

                if is_valid:
                    st.success(message)
                    with st.spinner("Detecting tables..."):
                        if page_image:
                            if processing_method == 'Computer Vision':
                                tables_info, steps = run_detection(
                                    page_image,
                                    model_choice,
                                    use_layout_analysis
                                )
                            else:
                                tables_info = detect_tables_pdfplumber(pdf_bytes, page_image)
                                steps = [("1. Original", page_image.copy())]
                            
                            if tables_info:
                                with st.spinner("Generating table names with AI..."):
                                    model, processor = load_paligemma_model_and_processor()
                                    
                                    for table in tables_info:
                                        bbox = table["location"]
                                        table_left = bbox["x0"]
                                        table_top = bbox["y0"]
                                        table_right = bbox["x1"]
                                        table_bottom = bbox["y1"]
                                        
                                        table_height = table_bottom - table_top
                                        header_height = int(table_height * 0.10)
                                        
                                        final_left = table_left
                                        final_right = table_right
                                      
                                        final_bottom = table_top + header_height

                                        extended_top = table_top - 400
                                        final_top = max(0, extended_top)
                                        
                                        final_crop_box = (
                                            final_left,
                                            final_top,
                                            final_right,
                                            final_bottom
                                        )
                                        
                                        final_crop_image = page_image.crop(final_crop_box)
                                        generated_name = generate_table_name(
                                            final_crop_image,
                                            model,
                                            processor
                                        )
                                        table["name"] = generated_name
                        
                                final_json_output = {
                                    "number_of_tables": len(tables_info),
                                    "tables": tables_info 
                                }
                                st.session_state.final_json = final_json_output
                            else:
                                st.session_state.final_json = {
                                    "number_of_tables": 0,
                                    "tables": []
                                }
                    
                            st.session_state.processing_steps = steps
                            st.session_state.results_ready = True
                        else:
                            st.session_state.results_ready = False
                    
    if 'results_ready' in st.session_state and st.session_state.results_ready:        
        st.subheader("Final Detection and Naming")
        
        final_data = st.session_state.final_json
        
        if final_data["number_of_tables"] > 0:
            st.success(f"Detected {final_data['number_of_tables']} table(s).")
            
            image_with_boxes = draw_bounding_boxes(
                st.session_state.page_image.copy(),
                final_data["tables"]
            )
            st.image(
                image_with_boxes,
                use_container_width=True,
                caption="Final detections with AI-generated names"
            )
            
            st.subheader("📋 Structured JSON Output")
            st.json(final_data)
        else:
            st.warning("No tables were detected.")
            st.image(st.session_state.page_image, use_container_width=True)

if __name__ == "__main__":
    main()
