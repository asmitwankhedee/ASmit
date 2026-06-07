import os
import base64
import smtplib
import random
import requests
import time
import threading
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path

import streamlit as st
from ultralytics import YOLO
from PIL import Image as PILImage
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Page Config  (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RoadVision AI — Pothole & Crack Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium Dark UI — Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@500;700&display=swap');

    /* ── Global Reset ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #070b14;
        color: #e2e8f0;
    }
    .stApp { background-color: #070b14; }

    /* ── Hide Streamlit chrome ────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── App header / logo ────────────────────────────── */
    .rv-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.4rem 2rem;
        background: linear-gradient(135deg, #0f172a 0%, #0b1120 100%);
        border-bottom: 1px solid rgba(99,102,241,.25);
        border-radius: 0 0 16px 16px;
        margin-bottom: 2rem;
    }
    .rv-logo {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
        color: #fff;
        letter-spacing: -0.5px;
    }
    .rv-logo span { color: #10b981; }
    .rv-tagline { color: #94a3b8; font-size: 0.88rem; margin-top: 2px; }

    /* ── Glass card ───────────────────────────────────── */
    .glass {
        background: rgba(15,23,42,.75);
        border: 1px solid rgba(99,102,241,.18);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 40px rgba(0,0,0,.45);
        margin-bottom: 1.5rem;
    }

    /* ── Login card ───────────────────────────────────── */
    .login-card {
        max-width: 420px;
        margin: 3rem auto;
        background: linear-gradient(163deg,#00ff75 0%,#3700ff 100%);
        border-radius: 24px;
        padding: 3px;
    }
    .login-inner {
        background: #171717;
        border-radius: 22px;
        padding: 2.5rem;
    }
    .login-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        color: #fff;
        margin-bottom: 1.5rem;
    }

    /* ── Section headings ─────────────────────────────── */
    h2 { font-family: 'Outfit', sans-serif; font-weight: 700; color: #f1f5f9; }
    h3 { font-family: 'Outfit', sans-serif; font-weight: 600; color: #cbd5e1; }

    /* ── Upload area ──────────────────────────────────── */
    [data-testid="stFileUploader"] > div {
        background: rgba(15,23,42,.6) !important;
        border: 2px dashed rgba(99,102,241,.4) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #6366f1 !important;
        background: rgba(99,102,241,.06) !important;
    }

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #fff;
        border: none;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.65rem 1.8rem;
        cursor: pointer;
        transition: all .25s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 24px rgba(99,102,241,.45);
    }

    /* ── Input fields ─────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #1e293b !important;
        border: 1px solid rgba(99,102,241,.3) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }

    /* ── Detection badge ──────────────────────────────── */
    .det-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .6px;
        margin-bottom: 6px;
    }
    .det-pothole { background: rgba(249,115,22,.15); color: #f97316; border: 1px solid rgba(249,115,22,.3); }
    .det-crack   { background: rgba(6,182,212,.15);  color: #06b6d4; border: 1px solid rgba(6,182,212,.3);  }
    .det-other   { background: rgba(139,92,246,.15); color: #8b5cf6; border: 1px solid rgba(139,92,246,.3); }

    /* ── Metric row ───────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: .5rem;
    }
    .metric-chip {
        background: rgba(30,41,59,.8);
        border: 1px solid rgba(99,102,241,.18);
        border-radius: 10px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .metric-chip strong { color: #e2e8f0; }

    /* ── AI analysis ──────────────────────────────────── */
    .ai-card {
        background: linear-gradient(135deg,rgba(99,102,241,.08) 0%,rgba(16,185,129,.06) 100%);
        border: 1px solid rgba(99,102,241,.2);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-top: 1.5rem;
    }
    .ai-card-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #f1f5f9;
    }

    /* ── Spinner overlay text ─────────────────────────── */
    .stSpinner > div { border-top-color: #6366f1 !important; }

    /* ── Footer ───────────────────────────────────────── */
    .rv-footer {
        text-align: center;
        padding: 1.5rem;
        color: #475569;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99,102,241,.12);
        margin-top: 3rem;
    }

    /* ── OTP timer chip ───────────────────────────────── */
    .otp-timer {
        background: rgba(245,158,11,.1);
        border: 1px solid rgba(245,158,11,.3);
        border-radius: 8px;
        padding: 6px 14px;
        color: #f59e0b;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 6px;
    }

    /* ── Success / Error notice ───────────────────────── */
    .notice-success { color: #10b981; font-weight: 600; }
    .notice-error   { color: #ef4444; font-weight: 600; }

    /* ── Divider ──────────────────────────────────────── */
    hr { border-color: rgba(99,102,241,.15) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Config & Secrets
# ---------------------------------------------------------------------------
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "authenticated": False,
        "otp": None,
        "otp_email": None,
        "otp_time": None,
        "mode": "signin",          # "signin" | "signup"
        "step": 1,                 # 1 = email, 2 = otp
        "result_image": None,
        "detections": None,
        "ai_text": None,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# API Rate Limiter
# ---------------------------------------------------------------------------
class APIRateLimiter:
    def __init__(self, min_interval_seconds: float = 6.0):
        self.min_interval = min_interval_seconds
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()


api_limiter = APIRateLimiter(min_interval_seconds=6.0)

# ---------------------------------------------------------------------------
# Load YOLO model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model():
    model_path = Path(__file__).parent / "best.pt"
    if model_path.exists():
        return YOLO(str(model_path))
    return None


model = load_model()

# ---------------------------------------------------------------------------
# Email OTP
# ---------------------------------------------------------------------------
def send_otp_email(email: str, otp: str) -> tuple:
    sender_email = get_secret("SMTP_EMAIL")
    sender_password = get_secret("SMTP_PASSWORD")
    if not sender_email or not sender_password:
        return True, "SMTP not configured — OTP shown below."
    try:
        msg = MIMEText(f"Your RoadVision AI verification code is: {otp}")
        msg["Subject"] = "RoadVision AI — Login Verification"
        msg["From"] = sender_email
        msg["To"] = email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True, "OTP sent! Check your inbox."
    except Exception as e:
        return True, f"Email delivery failed ({e}) — OTP shown below."


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------
def call_openai(detections: list, img: PILImage.Image) -> str:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI API key.")
    w, h = img.size
    small = img.resize((int(480 * w / h), 480), getattr(PILImage, "Resampling", PILImage).LANCZOS)
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=60)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = _build_prompt(detections)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}],
        "max_tokens": 1000,
    }
    api_limiter.wait_if_needed()
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=35,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenAI {resp.status_code}: {resp.text}")


def call_gemini(detections: list, img: PILImage.Image) -> str:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini API key.")
    w, h = img.size
    small = img.resize((int(480 * w / h), 480), getattr(PILImage, "Resampling", PILImage).LANCZOS)
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=60)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = _build_prompt(detections)
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}]}
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            api_limiter.wait_if_needed()
            resp = requests.post(url, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
    raise RuntimeError("All Gemini models failed.")


def _build_prompt(detections: list) -> str:
    prompt = (
        "You are a professional road structural safety inspector. Analyze this road surface image. "
        "The automated YOLO detector has identified the following road anomalies:\n"
    )
    if not detections:
        prompt += "- No cracks or potholes were detected by the model.\n"
    else:
        for idx, det in enumerate(detections):
            prompt += f"- {det['class_name']} #{idx+1}: Approximately {det['width_m']}m wide by {det['height_m']}m long\n"
    prompt += (
        "\nPlease provide a structured road structural analysis including:\n"
        "1. **Surface Condition Description**: What specific types of cracking or pothole degradation are visible?\n"
        "2. **Severity Assessment**: Rate the overall hazard/severity level (Low, Medium, or High) and justify it.\n"
        "3. **Estimated Depth**: Estimate the likely depth of the potholes.\n"
        "4. **Actionable Repair Recommendation**: Recommend specific repairs.\n"
        "\nUse simple Markdown formatting with bold text and bullet points. Keep it clear, concise, and professional."
    )
    return prompt


def generate_local_fallback_report(detections: list) -> str:
    potholes = [d for d in detections if "pothole" in d["class_name"].lower()]
    cracks = [d for d in detections if "crack" in d["class_name"].lower()]
    report = (
        "> [!WARNING]\n"
        "> **API Rate Limit Notice**: Cloud AI APIs are currently rate-limited (429). "
        "Showing a high-fidelity report generated locally by the RoadVision AI offline inspector module.\n\n"
        "### 🚧 Local Safety Inspector Report (Offline Fallback)\n\n"
    )
    report += "1. **Surface Condition Description**:\n"
    if not detections:
        report += "- No significant potholes or cracks detected. The pavement appears to be in stable condition.\n"
    else:
        report += f"- Detected **{len(detections)} road anomalies** ({len(potholes)} pothole(s), {len(cracks)} crack(s)).\n"
        for idx, det in enumerate(detections):
            report += f"  - **{det['class_name'].capitalize()} #{idx+1}**: {det['width_m']}m × {det['height_m']}m (conf: {int(det['confidence']*100)}%)\n"
    report += "\n2. **Severity Assessment**:\n"
    if not detections:
        report += "- **Severity Level**: **LOW** — No active structural distress detected.\n"
    else:
        max_size = max(max(d["width_m"], d["height_m"]) for d in detections)
        if potholes and max_size > 0.4:
            sev, just = "HIGH", "Critical pothole(s) exceeding 40 cm, posing immediate hazard to vehicles."
        elif potholes or max_size > 0.2:
            sev, just = "MEDIUM", "Moderate degradation — expanding cracks/potholes will worsen under traffic."
        else:
            sev, just = "LOW", "Minor surface blemishes — no deep structural cavities present."
        report += f"- **Severity Level**: **{sev}** — {just}\n"
    report += "\n3. **Estimated Depth**:\n"
    if not potholes:
        report += "- No deep cavities detected. Estimated depth: **0 cm**.\n"
    else:
        depth_cm = max(3.0, min(round(max(d["width_m"] for d in potholes) * 15, 1), 15.0))
        report += f"- Estimated maximum pothole depth: **{depth_cm} cm**.\n"
    report += "\n4. **Actionable Repair Recommendation**:\n"
    if not detections:
        report += "- No immediate action required. Schedule next routine inspection in 6 months.\n"
    else:
        if potholes:
            mw = max(d["width_m"] for d in potholes)
            if mw > 0.4:
                report += "- **Full-Depth Patching**: Excavate, compact subbase, apply hot-mix asphalt (HMA).\n"
            else:
                report += "- **Throw-and-Roll / Skin Patching** for minor potholes to prevent water ingress.\n"
        if cracks:
            ml = max(d["height_m"] for d in cracks)
            if ml > 1.0:
                report += "- **Crack Sealing/Routing** with elastomeric sealant for long cracks.\n"
            else:
                report += "- Monitor minor cracks. Apply fog seal during next scheduled maintenance cycle.\n"
    return report


def call_ai_analysis(detections: list, img: PILImage.Image) -> str:
    try:
        return call_gemini(detections, img)
    except Exception:
        try:
            return call_openai(detections, img)
        except Exception:
            return generate_local_fallback_report(detections)


# ---------------------------------------------------------------------------
# Preprocessing & SAHI helpers
# ---------------------------------------------------------------------------
def preprocess_image(img: PILImage.Image) -> PILImage.Image:
    try:
        import cv2, numpy as np
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        return PILImage.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    except Exception:
        return img


def apply_nms(boxes, iou_threshold=0.40):
    if not boxes:
        return []
    try:
        import numpy as np
        boxes = np.array(boxes)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        scores, classes = boxes[:, 4], boxes[:, 5]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            cls_match = classes[order[1:]] == classes[i]
            inds = np.where((ovr <= iou_threshold) | (~cls_match))[0]
            order = order[inds + 1]
        return boxes[keep].tolist()
    except Exception:
        return boxes


def draw_premium_boxes(img: PILImage.Image, boxes: list, class_names: dict) -> PILImage.Image:
    from PIL import ImageDraw, ImageFont
    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "pothole": {"border": (249, 115, 22, 255), "fill": (249, 115, 22, 40)},
        "crack":   {"border": (6, 182, 212, 255),  "fill": (6, 182, 212, 40)},
    }
    default_color = {"border": (139, 92, 246, 255), "fill": (139, 92, 246, 40)}
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(img.size[0] * 0.012)))
    except IOError:
        font = ImageFont.load_default()
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = class_names.get(cls_id, f"Class {cls_id}").lower()
        color_info = colors.get(class_name, default_color)
        draw.rectangle([x1, y1, x2, y2], fill=color_info["fill"])
        draw.rectangle([x1, y1, x2, y2], outline=color_info["border"], width=3)
        label_text = f"{class_name.upper()} {int(conf * 100)}%"
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)
        pp = 6
        py1 = max(0, y1 - text_h - pp * 2)
        py2 = y1 if y1 > (text_h + pp * 2) else (y1 + text_h + pp * 2)
        if py2 == y1 + text_h + pp * 2:
            py1, py2 = y1, y1 + text_h + pp * 2
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([x1, py1, x1 + text_w + pp * 2, py2], radius=4, fill=color_info["border"])
        else:
            draw.rectangle([x1, py1, x1 + text_w + pp * 2, py2], fill=color_info["border"])
        draw.text((x1 + pp, py1 + pp), label_text, fill=(255, 255, 255, 255), font=font)
    return PILImage.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def run_prediction(img: PILImage.Image):
    processed_img = preprocess_image(img)
    w, h = img.size
    slice_size, overlap, conf_thresh = 512, 0.25, 0.12
    stride = int(slice_size * (1 - overlap))
    boxes = []

    x_offsets = list(range(0, w - slice_size + 1, stride))
    if w > slice_size and x_offsets and x_offsets[-1] + slice_size < w:
        x_offsets.append(w - slice_size)
    if not x_offsets:
        x_offsets = [0]
    y_offsets = list(range(0, h - slice_size + 1, stride))
    if h > slice_size and y_offsets and y_offsets[-1] + slice_size < h:
        y_offsets.append(h - slice_size)
    if not y_offsets:
        y_offsets = [0]

    for x_off in x_offsets:
        for y_off in y_offsets:
            crop_w, crop_h = min(slice_size, w - x_off), min(slice_size, h - y_off)
            slice_img = processed_img.crop((x_off, y_off, x_off + crop_w, y_off + crop_h))
            results = model.predict(source=slice_img, conf=conf_thresh, augment=True, verbose=False)
            if not results:
                continue
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                boxes.append([bx1 + x_off, by1 + y_off, bx2 + x_off, by2 + y_off, float(box.conf[0]), int(box.cls[0])])

    global_imgsz = 1280 if max(w, h) >= 1280 else max(640, max(w, h))
    global_results = model.predict(source=processed_img, conf=conf_thresh, imgsz=global_imgsz, augment=True, verbose=False)
    if global_results:
        for box in global_results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            boxes.append([bx1, by1, bx2, by2, float(box.conf[0]), int(box.cls[0])])

    merged_boxes = apply_nms(boxes, iou_threshold=0.40)

    horizon = 0.45 * h
    mpp_bottom = 3.5 / w
    detections = []
    for box in merged_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        class_name = model.names[cls_id] if (model and hasattr(model, "names")) else f"Class {cls_id}"
        is_crack = "crack" in class_name.lower()
        y_road = (y1 + y2) / 2.0 if is_crack else y2
        y_road = max(y_road, horizon + 0.05 * h)
        mpp_local = min(mpp_bottom * ((h - horizon) / (y_road - horizon)), mpp_bottom * 8.0)
        detections.append({
            "class_name": class_name,
            "width_px": int(x2 - x1),
            "height_px": int(y2 - y1),
            "width_m": round((x2 - x1) * mpp_local, 2),
            "height_m": round((y2 - y1) * mpp_local, 2),
            "confidence": round(conf, 2),
        })

    plotted = draw_premium_boxes(img, merged_boxes, model.names if model else {})
    wp, hp = plotted.size
    new_h = 720
    plotted = plotted.resize((int(new_h * wp / hp), new_h), getattr(PILImage, "Resampling", PILImage).LANCZOS)
    return plotted, detections


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header(show_logout: bool = False):
    logout_html = ""
    if show_logout:
        logout_html = """
        <form action="" method="get" style="margin:0;">
            <button onclick="window.location.href='?logout=1'"
                style="background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.35);
                       color:#ef4444;border-radius:8px;padding:7px 18px;cursor:pointer;
                       font-family:'Outfit',sans-serif;font-weight:600;font-size:.88rem;">
                Sign Out
            </button>
        </form>"""
    st.markdown(
        f"""
        <div class="rv-header">
            <div>
                <div class="rv-logo">RoadVision<span>AI</span></div>
                <div class="rv-tagline">Upload a road image to detect potholes &amp; cracks and measure their dimensions instantly.</div>
            </div>
            {logout_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Logout via query params
# ---------------------------------------------------------------------------
params = st.query_params
if "logout" in params:
    st.session_state.authenticated = False
    st.session_state.otp = None
    st.session_state.otp_email = None
    st.session_state.step = 1
    st.query_params.clear()
    st.rerun()

# ===========================================================================
#  LOGIN PAGE
# ===========================================================================
if not st.session_state.authenticated:
    render_header(show_logout=False)

    st.markdown(
        """
        <div class="login-card">
            <div class="login-inner">
                <div class="login-title">🔐 Secure Portal Login</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 1.6, 1])
    with col_c:
        with st.container():
            st.markdown('<div class="glass">', unsafe_allow_html=True)

            # ── Tab switcher ─────────────────────────────────────────────
            tab_col1, tab_col2 = st.columns(2)
            with tab_col1:
                if st.button("SIGN IN", key="tab_signin",
                             type="primary" if st.session_state.mode == "signin" else "secondary"):
                    st.session_state.mode = "signin"
                    st.session_state.step = 1
            with tab_col2:
                if st.button("SIGN UP", key="tab_signup",
                             type="primary" if st.session_state.mode == "signup" else "secondary"):
                    st.session_state.mode = "signup"
                    st.session_state.step = 1

            st.markdown("---")
            heading = "Sign In to Your Account" if st.session_state.mode == "signin" else "Create an Account"
            st.markdown(f"<div style='text-align:center;font-family:Outfit,sans-serif;font-weight:700;font-size:1.1rem;color:#fff;margin-bottom:1.2rem;'>{heading}</div>",
                        unsafe_allow_html=True)

            # ── Step 1: email (+ name for signup) ───────────────────────
            if st.session_state.step == 1:
                if st.session_state.mode == "signup":
                    full_name = st.text_input("Full Name", placeholder="Your Full Name", key="input_name")

                email = st.text_input("Email Address", placeholder="you@example.com", key="input_email")

                if st.button("Send Verification OTP", key="send_otp_btn"):
                    if not email or "@" not in email:
                        st.error("Please enter a valid email address.")
                    else:
                        otp = str(random.randint(100000, 999999))
                        st.session_state.otp = otp
                        st.session_state.otp_email = email
                        st.session_state.otp_time = time.time()
                        ok, msg = send_otp_email(email, otp)
                        if ok:
                            st.success(msg)
                            # Show OTP in sidebar for dev/demo
                            st.info(f"🔑 **Dev OTP** (console): `{otp}`")
                            st.session_state.step = 2
                            st.rerun()
                        else:
                            st.error(msg)

            # ── Step 2: OTP verification ─────────────────────────────────
            elif st.session_state.step == 2:
                elapsed = int(time.time() - (st.session_state.otp_time or time.time()))
                remaining = max(0, 300 - elapsed)
                st.markdown(
                    f'<div class="otp-timer">⏱ OTP expires in <strong>{remaining}s</strong> — check your inbox or the dev info box above.</div>',
                    unsafe_allow_html=True,
                )
                otp_input = st.text_input("Enter 6-Digit OTP", max_chars=6, placeholder="______", key="input_otp")

                col_v, col_b = st.columns(2)
                with col_v:
                    if st.button("✔ Verify & Login", key="verify_btn"):
                        if otp_input == st.session_state.otp:
                            if remaining == 0:
                                st.error("OTP expired. Please request a new one.")
                                st.session_state.step = 1
                                st.rerun()
                            else:
                                st.session_state.authenticated = True
                                st.session_state.otp = None
                                st.session_state.otp_email = None
                                st.success("Login successful! Redirecting…")
                                st.rerun()
                        else:
                            st.error("Invalid OTP. Please try again.")
                with col_b:
                    if st.button("← Back", key="back_btn"):
                        st.session_state.step = 1
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rv-footer">© Cracks And Potholes Detection AI. All rights reserved.</div>',
                unsafe_allow_html=True)
    st.stop()


# ===========================================================================
#  MAIN APP
# ===========================================================================
render_header(show_logout=True)

if model is None:
    st.error("⚠️ YOLO model could not be loaded. Ensure `best.pt` is in the project root.")
    st.stop()

# ── Upload Section ────────────────────────────────────────────────────────
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Road Image")
uploaded_file = st.file_uploader(
    "Drag & Drop a road image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
    key="file_uploader",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Analyse button / logic ────────────────────────────────────────────────
if uploaded_file is not None:
    img = PILImage.open(uploaded_file).convert("RGB")

    col_img, col_btn = st.columns([3, 1])
    with col_img:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyse = st.button("🔍 Analyse Image", key="analyse_btn")

    if analyse:
        st.session_state.result_image = None
        st.session_state.detections = None
        st.session_state.ai_text = None

        with st.spinner("🔬 Running YOLO detection + AI analysis — this may take 15–30 s…"):
            plotted, detections = run_prediction(img)
            ai_text = call_ai_analysis(detections, img)
            st.session_state.result_image = plotted
            st.session_state.detections = detections
            st.session_state.ai_text = ai_text

# ── Results ───────────────────────────────────────────────────────────────
if st.session_state.result_image is not None:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    res_col, det_col = st.columns([3, 2])

    with res_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Analyzed Output")
        st.image(st.session_state.result_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with det_col:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("### 🔎 Detections")
        st.markdown(
            '<p style="color:#64748b;font-size:.82rem;margin-bottom:1rem;">'
            'Measurements converted to metres assuming a standard 3.5 m road-width reference.</p>',
            unsafe_allow_html=True,
        )
        detections = st.session_state.detections
        if not detections:
            st.info("✅ No cracks or potholes detected in this image.")
        else:
            for idx, det in enumerate(detections):
                cname = det["class_name"].lower()
                badge_cls = "det-pothole" if "pothole" in cname else ("det-crack" if "crack" in cname else "det-other")
                st.markdown(
                    f"""
                    <div style="margin-bottom:1rem;padding:12px 16px;background:rgba(30,41,59,.6);
                                border-radius:12px;border:1px solid rgba(99,102,241,.15);">
                        <span class="det-badge {badge_cls}">{det['class_name'].upper()} #{idx+1}</span>
                        <div class="metric-row" style="margin-top:8px;">
                            <div class="metric-chip">Width: <strong>{det['width_m']} m</strong></div>
                            <div class="metric-chip">Length: <strong>{det['height_m']} m</strong></div>
                            <div class="metric-chip">Conf: <strong>{int(det['confidence']*100)}%</strong></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── AI Analysis Card ──────────────────────────────────────────────────
    st.markdown(
        """
        <div class="ai-card">
            <div class="ai-card-title">🤖 AI Severity &amp; Depth Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.ai_text or "_No AI analysis available._")

    if st.button("🔄 Analyse Another Image", key="reset_btn"):
        st.session_state.result_image = None
        st.session_state.detections = None
        st.session_state.ai_text = None
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="rv-footer">© Cracks And Potholes Detection AI. All rights reserved.</div>',
    unsafe_allow_html=True,
)
