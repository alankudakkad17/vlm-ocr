!pip uninstall -y paddlepaddle paddlepaddle-gpu langchain
!pip install -q easyocr transformers==4.49.0 pymupdf einops timm accelerate bitsandbytes

import fitz  # PyMuPDF
import torch
import io
import requests
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import easyocr

# ==========================================
# 1. CONFIGURATION & INITIALIZATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Initializing models on {device}. This may take a minute...\n")

# Initialize VLM (Florence-2) for Layout Analysis 
print("Loading Florence-2 (Layout Analyzer)...")
vlm_id = "microsoft/Florence-2-large"
processor = AutoProcessor.from_pretrained(vlm_id, trust_remote_code=True)
vlm_model = AutoModelForCausalLM.from_pretrained(
    vlm_id,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

vlm_model.config.decoder_start_token_id = 2
vlm_model.config.pad_token_id = 1

# Initialize Traditional OCR (EasyOCR) for Text Extraction 
print("Loading EasyOCR (Text Extractor)...")
ocr_engine = easyocr.Reader(['en'], gpu=True if device == "cuda" else False, verbose=False)

# Initialize Local LLM (Qwen2.5-3B) for Contextual Correction 
print("Loading Qwen2.5-3B (Contextual Reasoner)...")
llm_id = "Qwen/Qwen2.5-3B-Instruct"

# 4-bit Quantization config to prevent Out-Of-Memory crashes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch_dtype
)

llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    quantization_config=bnb_config,
    device_map="auto" 
)

# ==========================================
# 2. CORE PIPELINE FUNCTIONS
# ==========================================

def get_layout_from_vlm(image):
    """Uses Florence-2 to interpret the page and find text regions."""
    task_prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = vlm_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer[task_prompt]

def extract_text_with_easyocr(image, layout_data):
    """Crops regions based on VLM coordinates and runs EasyOCR."""
    extracted_regions = []

    if 'quad_boxes' not in layout_data or 'labels' not in layout_data:
        return extracted_regions

    for box, vlm_text in zip(layout_data['quad_boxes'], layout_data['labels']):
        x_coords, y_coords = box[0::2], box[1::2]
        crop_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


        buffer = 2
        crop_box = (
            max(0, crop_box[0] - buffer),
            max(0, crop_box[1] - buffer),
            min(image.width, crop_box[2] + buffer),
            min(image.height, crop_box[3] + buffer)
        )

        crop = image.crop(crop_box)
        ocr_result = ocr_engine.readtext(np.array(crop))

        final_text = vlm_text
        if ocr_result:
            # Join all text found in this specific crop
            final_text = " ".join([res[1] for res in ocr_result])

        extracted_regions.append({
            "box": crop_box,
            "text": final_text
        })

    return extracted_regions

def correct_and_reconstruct_with_llm(extracted_regions):
    """Passes spatial data and raw text to Qwen for intelligent reconstruction."""
    if not extracted_regions:
        return "No text regions detected."


    messages = [
        {"role": "system", "content": "You are an expert document reconstruction AI. I am providing you with a JSON list containing text snippets and their spatial bounding box coordinates (left, top, right, bottom) extracted from a document. Your task is to analyze the coordinates to understand the physical layout, correct any spelling errors caused by the OCR engine, and output the final document in cleanly formatted Markdown. Do not include any explanations, only return the Markdown text."},
        {"role": "user", "content": f"Input Data:\n{json.dumps(extracted_regions, indent=2)}"}
    ]

    prompt_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = llm_tokenizer(prompt_text, return_tensors="pt").to(device)


    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.1,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )


    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response

# ==========================================
# 3. MAIN ORCHESTRATOR
# ==========================================

def process_document(pdf_path_or_url, render_dpi=150):
    """Orchestrates the full pipeline across a PDF document."""
    if pdf_path_or_url.startswith('http'):
        response = requests.get(pdf_path_or_url)
        response.raise_for_status()
        pdf_data = io.BytesIO(response.content)
    else:
        with open(pdf_path_or_url, "rb") as f:
            pdf_data = io.BytesIO(f.read())

    doc = fitz.open(stream=pdf_data, filetype="pdf")
    final_document_markdown = []

    print(f"\nProcessing {len(doc)} pages...")

    for page_num in range(len(doc)):
        print(f"\n--- Analyzing Page {page_num + 1} ---")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(render_dpi/72, render_dpi/72))
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        print("1. Extracting layout with Florence-2...")
        layout_data = get_layout_from_vlm(image)

        print("2. Reading text regions with EasyOCR...")
        extracted_regions = extract_text_with_easyocr(image, layout_data)

        print("3. Reconstructing and correcting with Qwen2.5-3B...")
        page_markdown = correct_and_reconstruct_with_llm(extracted_regions)

        print("\n[Final Page Output]")
        print(page_markdown)

        final_document_markdown.append(page_markdown)

    doc.close()
    return "\n\n---\n\n".join(final_document_markdown)

# ==========================================
# EXECUTION
# ==========================================
# Uncomment these lines and paste a PDF URL to test it!
test_url = "11.pdf"
final_result = process_document(test_url)

!pip install -q jiwer

import fitz # PyMuPDF
import jiwer
import re
import string

def normalize_text(text):
    """
    Cleans text for a fair OCR evaluation.
    Removes punctuation, lowercases everything, and normalizes spacing.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_ground_truth_from_pdf(pdf_path):
    """Extracts the perfect text layer from a digital reference PDF."""
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        # Extract text directly from the digital layer, no OCR
        text = page.get_text()
        full_text.append(text)

    doc.close()
    return "\n".join(full_text)

def evaluate_pipeline(predicted_text, reference_pdf_path):
    """Compares the pipeline's output against the reference PDF."""

    print("Extracting ground truth from reference PDF...")
    ground_truth_text = extract_ground_truth_from_pdf(reference_pdf_path)

    print("Normalizing texts for fair comparison...")
    clean_predicted = normalize_text(predicted_text)
    clean_ground_truth = normalize_text(ground_truth_text)

    # Calculate Metrics
    wer = jiwer.wer(clean_ground_truth, clean_predicted)
    cer = jiwer.cer(clean_ground_truth, clean_predicted)

    print("\n" + "="*40)
    print("🎯 EVALUATION RESULTS 🎯")
    print("="*40)
    # Convert to percentages for easier reading
    print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print("="*40)

    # If the error is high, it helps to see exactly what went wrong
    if wer > 0.0:
        print("\n[Snippet Comparison - First 200 chars]")
        print(f"Ground Truth: {clean_ground_truth[:200]}...")
        print(f"Predicted:    {clean_predicted[:200]}...")

    return {"wer": wer, "cer": cer}

# ==========================================
# EXECUTION
# ==========================================
# Assuming 'final_document_markdown' is the string output from your Qwen pipeline
final_document_markdown = process_document(final_result)

reference_pdf_path = "reference.pdf"
metrics = evaluate_pipeline(final_document_markdown, reference_pdf_path)
