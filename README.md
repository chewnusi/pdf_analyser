# PDF Analyzer

## Project Description

This project is a visual layout-aware PDF table detection application built with Streamlit. It allows users to upload PDF documents and analyze them to detect tables using various models. The application supports multiple table detection models, including TableTransformer, Florence-2, YOLOv8, and a text-based approach using pdfplumber. It also features layout analysis for improved detection accuracy and an AI-powered table naming functionality using the PaliGemma model.

## Prerequisites

To run this project, you need to have the following installed:

- Python 3.9+
- Docker and Docker Compose (for containerized setup)
- A Hugging Face account and an access token (for downloading pre-trained models).
- PaliGemma Model: To use the AI-powered table naming feature, you must visit the [PaliGemma model page](https://huggingface.co/google/paligemma-3b-pt-224) on Hugging Face and accept its terms of usage before running the application.
- GPU is recommended for faster processing, but not mandatory.

The project also depends on several Python libraries, which are listed in the `requirements.txt` file. These include:

- streamlit
- PyPDF2
- Pillow
- numpy
- PyMuPDF
- torch
- transformers==4.49.0
- ultralytics
- huggingface-hub
- pytesseract
- scikit-learn
- pdfplumber
- timm
- accelerate>=0.26.0
- einops


## Docker Usage

For a containerized setup, you can use the provided Docker configuration.

1.  **Set up the environment:**
    This will create a `.env` file from the template. You need to edit this file to add your Hugging Face token.
    ```bash
    make setup
    ```

3.  **Build the Docker image:**
    ```bash
    make build
    ```

4.  **Run the application:**
    ```bash
    make up
    ```
    The application will be available at: [http://localhost:8501](http://localhost:8501)

    > **Note:** The first time you select a model for table detection, it may take longer to start as the model needs to be downloaded. Subsequent uses will be faster.

5.  **View logs (optional):**
    ```bash
    make logs
    ```

6.  **Shut down the application:**
    ```bash
    make down
    ```

## Setup Instructions

If you prefer not to use Docker, you can set up the environment locally:

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Login to Hugging Face:**
    ```bash
    hf auth login
    ```

4.  **Run the application:**
    ```bash
    streamlit run main.py
    ```

    > **Note:** The first time you select a model for table detection, it may take longer to start as the model needs to be downloaded. Subsequent uses will be faster.

## Limitations and Future Improvements

### Current Limitations
*   **Table Detection Accuracy:** The computer vision models (TableTransformer, YOLOv8, Florence-2) can sometimes struggle to distinguish between closely spaced tables, potentially merging them into a single detection.
*   **Table Naming:** While PaliGemma provides a powerful open-source solution for naming tables, proprietary models (like GPT-4o or Gemini) might offer higher accuracy and generate more contextually relevant names.
*   **Performance:** Processing can be slow, especially on CPUs, due to the size and complexity of the deep learning models. The initial download and loading of models also adds to the startup time for each model type.
*   **Complex Layouts:** The application may not perform well on PDFs with highly complex or unconventional layouts, such as multi-page tables or tables embedded within other complex visual elements.
*   **Scanned Documents:** The accuracy for scanned (image-based) PDFs is highly dependent on the quality of the scan and the underlying OCR capabilities, which might not be as robust as text-based PDF processing.

### Alternative Approaches

- **LLM with Unstructured for Table Naming**: Use a library like Unstructured.io to preprocess a document image, isolating elements like titles and headers. This text can then be fed to a Large Language Model (LLM) like GPT or Gemini to generate a contextually aware name for the detected table.
- **Camelot for Table Detection**: Employ Camelot for fast and precise table extraction from digitally native PDFs. It excels at parsing tables defined by either clear grid lines or consistent whitespace alignment but is not suitable for scanned (image-based) documents or tables with highly unconventional layouts.
- **Classical CV Methods for Image-Based Detection**: For scanned documents, apply classical computer vision techniques. Methods like Line Detection (using Hough Transform), Contour Detection (finding cell shapes), or Projection Profiling (analyzing pixel histograms) can identify table structures directly from the image.

## Example of work

<video src="example/coxit-test.mp4" controls="controls" style="max-width: 720px;"></video>
