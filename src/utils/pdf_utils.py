import io
import PyPDF2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import fitz

def validate_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        if len(pdf_reader.pages) == 1:
            uploaded_file.seek(0)
            return True, "✅ PDF validation successful!"
        else:
            return False, f"❌ PDF must contain exactly 1 page. Yours has {len(pdf_reader.pages)}."
    except Exception as e:
        return False, f"❌ Error reading PDF: {str(e)}"

def pdf_to_image(pdf_bytes, dpi=300):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as e:
        st.error(f"Error converting PDF to image: {e}")
        return None

def draw_bounding_boxes(image, tables_info):
    draw = ImageDraw.Draw(image) 
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
        
    colors = ["red", "green", "blue", "orange", "purple", "brown"]
    for idx, table in enumerate(tables_info):
        bbox = table["location"]
        name = table.get("name", "Unnamed Table")
        color = colors[idx % len(colors)]
        
        draw.rectangle([bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]], outline=color, width=8)
        
        text_position = (bbox["x0"] + 5, bbox["y0"] + 5)
        text_bbox = draw.textbbox(text_position, name, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text(text_position, name, fill="white", font=font)

    return image
