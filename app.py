import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# 🌱 Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 🧾 File paths
CSV_LOG_FILE = "vlm_training_data.csv"
JSONL_LOG_FILE = "vlm_training_data.jsonl"

# 🎨 Page config
st.set_page_config(page_title="VLM Autonomous Driving MVP", layout="centered")
st.title("🚘 Vision-Language Autonomous Driving Assistant (Gemini)")

# ✅ Log data to CSV
def log_to_csv(image_name, prompt, completion, hazard_tag="No"):
    file_exists = os.path.isfile(CSV_LOG_FILE)
    with open(CSV_LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "image_filename", "prompt", "completion", "hazard_tag"])
        writer.writerow([datetime.now().isoformat(), image_name, prompt, completion, hazard_tag])

# ✅ Convert CSV to JSONL
def convert_csv_to_jsonl():
    if os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            with open(JSONL_LOG_FILE, 'w', encoding='utf-8') as jsonlfile:
                for row in reader:
                    json_obj = {
                        "image": row["image_filename"],
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                        "hazard_tag": row["hazard_tag"]
                    }
                    jsonlfile.write(json.dumps(json_obj) + "\n")

# 🧠 Analyze image using Gemini
def analyze_image_with_gemini(image, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt, image], stream=False)

    result = response.text
    hazard_tag = "Yes" if any(k in result.lower() for k in ["accident", "hazard", "danger", "collision", "crash"]) else "No"
    return result, hazard_tag

# 🔘 User input mode
input_mode = st.radio("Choose Input Mode", ["Upload Image", "Capture with Webcam", "Dashcam Mode"])
image = None

# 📤 Upload mode
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload a driving scene image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# 📸 Webcam capture
elif input_mode == "Capture with Webcam":
    if st.button("📸 Capture Image from Webcam"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            st.error("Failed to capture from webcam.")

# 🎥 Dashcam mode
elif input_mode == "Dashcam Mode":
    video_file = st.file_uploader("🎥 Upload Dashcam Footage", type=["mp4", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = 10 * fps  # Every 10 seconds
        current_frame = 0
        frame_count = 0

        st.info("⏳ Processing video frames every 10 seconds...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                prompt = "Analyze the attached driving scene image and suggest what action should be taken."
                result, hazard_tag = analyze_image_with_gemini(pil_img, prompt)

                filename = f"frame_{frame_count}.jpg"
                pil_img.save(filename)
                log_to_csv(filename, prompt, result, hazard_tag)

                frame_count += 1
            current_frame += 1

        cap.release()
        st.success(f"✅ Processed {frame_count} frames. Check CSV for details.")

# 🧠 Manual image analysis
if image:
    st.image(image, caption="Selected Driving Scene", use_column_width=True)
    if st.button("🔍 Analyze"):
        prompt = "Analyze the attached driving scene image and suggest what action should be taken."
        result, hazard_tag = analyze_image_with_gemini(image, prompt)

        filename = "uploaded_image.png"
        image.save(filename)
        log_to_csv(filename, prompt, result, hazard_tag)

        # Add emojis based on result
        emojis = {
            "accident": "💥",
            "hazard": "⚠️",
            "collision": "🚨",
            "danger": "🔥",
            "crash": "🛑",
            "safe": "🛣️",
            "clear": "✅"
        }
        for keyword, emoji in emojis.items():
            if keyword in result.lower():
                result = f"{emoji} {result}"
                break  # Only use the first matching emoji

        # Load Google Font and Style
        st.markdown("""
            <link href="https://fonts.googleapis.com/css2?family=Fira+Code&display=swap" rel="stylesheet">
            <style>
                .gemini-box {
                    font-family: 'Fira Code', monospace;
                    background-color: #f8f9fa;
                    border-left: 6px solid #1e88e5;
                    padding: 20px;
                    margin-top: 15px;
                    border-radius: 10px;
                    color: #212121;
                    line-height: 1.6;
                    font-size: 15px;
                    white-space: pre-wrap;
                }
            </style>
        """, unsafe_allow_html=True)

        st.success("✅ Scene Analyzed")
        st.markdown("### 🧾 TAARP ANALYSIS")
        st.markdown(f"<div class='gemini-box'>{result}</div>", unsafe_allow_html=True)

# 🔄 Convert CSV to JSONL
if st.button("🔁 Convert to JSONL for Fine-tuning"):
    with st.spinner("Converting to JSONL..."):
        convert_csv_to_jsonl()
    st.success("✅ JSONL file ready for download!")

# 📥 Download buttons
if os.path.exists(JSONL_LOG_FILE):
    with open(JSONL_LOG_FILE, "rb") as f:
        st.download_button("📥 Download Fine-tuning Dataset (JSONL)", f, file_name="vlm_training_data.jsonl")

if os.path.exists(CSV_LOG_FILE):
    with open(CSV_LOG_FILE, "rb") as f:
        st.download_button("📥 Download CSV Log", f, file_name="vlm_training_data.csv")

# Footer
st.markdown("---")
st.caption("Final Year Project by Tushar And Amrit Raj Paramhans")
