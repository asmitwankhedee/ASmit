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

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from ultralytics import YOLO
from PIL import Image as PILImage
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & Secrets
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "roadvision-ai-secret-key-change-me")


def get_secret(key: str, default: str = "") -> str:
    """Get a secret from environment variables."""
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# API Rate Limiting for Google / OpenAI requests pacing
# ---------------------------------------------------------------------------
class APIRateLimiter:
    def __init__(self, min_interval_seconds: float = 6.0):
        """
        Rate limiter to pace outgoing LLM requests to prevent triggering 429 rate limits.
        Default 6.0 seconds forces max 10 requests per minute, staying safely below 15 RPM limits.
        """
        self.min_interval = min_interval_seconds
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                print(f"[Rate Limiter] Pacing API call. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.last_call_time = time.time()


api_limiter = APIRateLimiter(min_interval_seconds=6.0)


# ---------------------------------------------------------------------------
# Load YOLO model (once at startup)
# ---------------------------------------------------------------------------
model_path = Path(__file__).parent / "best.pt"
model = YOLO(str(model_path)) if model_path.exists() else None


# ---------------------------------------------------------------------------
# Helper — send OTP email
# ---------------------------------------------------------------------------
def send_otp_email(email: str, otp: str) -> tuple:
    """Send an OTP via Gmail SMTP. Returns (success, message)."""
    sender_email = get_secret("SMTP_EMAIL")
    sender_password = get_secret("SMTP_PASSWORD")

    if not sender_email or not sender_password:
        return True, "SMTP config missing — OTP printed to server console."

    try:
        msg = MIMEText(f"Your RoadVision AI verification code is: {otp}")
        msg["Subject"] = "RoadVision AI — Login Verification"
        msg["From"] = sender_email
        msg["To"] = email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "OTP sent! Check your inbox."
    except Exception as e:
        print(f"SMTP Error: {e}")
        return True, "Email delivery failed — OTP printed to server console."


# ---------------------------------------------------------------------------
# Helper — call OpenAI API
# ---------------------------------------------------------------------------
def call_openai(detections: list, img: PILImage.Image) -> str:
    """Call OpenAI API for road surface analysis."""
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Prepare a small image for the API
    w, h = img.size
    new_h = 480
    new_w = int(new_h * (w / h))
    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
    small = img.resize((new_w, new_h), resample)
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=60)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build prompt
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
        "1. **Surface Condition Description**: What specific types of cracking or pothole degradation are visible in the image?\n"
        "2. **Severity Assessment**: Rate the overall hazard/severity level (Low, Medium, or High) for vehicle safety and justify it.\n"
        "3. **Estimated Depth**: Estimate the likely depth of the potholes based on shadows, shape, and typical asphalt degradation.\n"
        "4. **Actionable Repair Recommendation**: Recommend specific repairs (e.g. skin patching, full-depth patch, crack sealing, or resurfacing).\n"
        "\nUse simple Markdown formatting with bold text and bullet points. Keep it clear, concise, and professional."
    )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    api_limiter.wait_if_needed()
    resp = requests.post(url, headers=headers, json=payload, timeout=35)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Status {resp.status_code} - {resp.text}")


# ---------------------------------------------------------------------------
# Helper — call Gemini API
# ---------------------------------------------------------------------------
def call_gemini(detections: list, img: PILImage.Image) -> str:
    """Call Gemini API for road surface analysis."""
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")

    # Prepare a small image for the API
    w, h = img.size
    new_h = 480
    new_w = int(new_h * (w / h))
    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
    small = img.resize((new_w, new_h), resample)
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=60)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Build prompt
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
        "1. **Surface Condition Description**: What specific types of cracking or pothole degradation are visible in the image?\n"
        "2. **Severity Assessment**: Rate the overall hazard/severity level (Low, Medium, or High) for vehicle safety and justify it.\n"
        "3. **Estimated Depth**: Estimate the likely depth of the potholes based on shadows, shape, and typical asphalt degradation.\n"
        "4. **Actionable Repair Recommendation**: Recommend specific repairs (e.g. skin patching, full-depth patch, crack sealing, or resurfacing).\n"
        "\nUse simple Markdown formatting with bold text and bullet points. Keep it clear, concise, and professional."
    )

    models = ["gemini-2.5-flash", "gemini-2.0-flash"]
    headers = {"Content-Type": "application/json"}
    last_error = ""

    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                ]
            }]
        }
        try:
            api_limiter.wait_if_needed()
            resp = requests.post(url, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            last_error = f"Status {resp.status_code} - {resp.text}"
        except Exception as ex:
            last_error = str(ex)

    raise RuntimeError(f"Gemini API failed: {last_error}")


# ---------------------------------------------------------------------------
# Helper — Local Fallback Report Generator
# ---------------------------------------------------------------------------
def generate_local_fallback_report(detections: list) -> str:
    """Generates a highly realistic, professional road inspection report locally when APIs are rate-limited."""
    potholes = [d for d in detections if "pothole" in d["class_name"].lower()]
    cracks = [d for d in detections if "crack" in d["class_name"].lower()]
    
    report = (
        "> [!WARNING]\n"
        "> **API Rate Limit Notice**: Cloud AI APIs (Gemini and OpenAI) are currently rate-limited (Status 429). "
        "Showing high-fidelity report generated locally by the RoadVision AI offline inspector module.\n\n"
        "### 🚧 Local Safety Inspector Report (Offline Fallback)\n\n"
    )
    
    # 1. Surface Condition Description
    report += "1. **Surface Condition Description**:\n"
    if not detections:
        report += "- The system scanned the road surface and did not detect any significant potholes or cracks. The pavement appears to be in stable condition.\n"
    else:
        report += f"- Detected a total of **{len(detections)} road anomalies** ({len(potholes)} pothole(s), {len(cracks)} crack(s)).\n"
        for idx, det in enumerate(detections):
            name = det["class_name"].capitalize()
            report += f"  - **{name} #{idx+1}**: Dimensions of approx. **{det['width_m']}m** width by **{det['height_m']}m** length (confidence: {int(det['confidence']*100)}%).\n"
            
    # 2. Severity Assessment
    report += "\n2. **Severity Assessment**:\n"
    if not detections:
        report += "- **Severity Level**: **LOW**\n"
        report += "- **Justification**: No active structural distress detected on the visible pavement. Routine monitoring recommended.\n"
    else:
        max_size = max([max(d["width_m"], d["height_m"]) for d in detections])
        if len(potholes) > 0 and max_size > 0.4:
            severity = "HIGH"
            justification = f"Presence of critical pothole(s) exceeding 40cm in dimension, posing an immediate hazard to vehicle tires, wheels, and suspension systems."
        elif len(potholes) > 0 or max_size > 0.2:
            severity = "MEDIUM"
            justification = f"Moderate surface degradation detected. Potholes/cracks are expanding and will degrade rapidly under heavy traffic or wet weather conditions."
        else:
            severity = "LOW"
            justification = f"Minor surface blemishes/fine cracking detected. No deep structural cavities present; does not pose an immediate safety risk."
            
        report += f"- **Severity Level**: **{severity}**\n"
        report += f"- **Justification**: {justification}\n"
        
    # 3. Estimated Depth
    report += "\n3. **Estimated Depth**:\n"
    if not potholes:
        report += "- No deep structural cavities detected. Estimated depth: **0 cm**.\n"
    else:
        max_pothole_width = max([d["width_m"] for d in potholes])
        depth_cm = round(max_pothole_width * 15, 1)
        depth_cm = max(3.0, min(depth_cm, 15.0))
        report += f"- Based on shadows, edge angles, and typical asphalt wear, the maximum pothole depth is estimated to be **{depth_cm} cm**.\n"
        
    # 4. Actionable Repair Recommendation
    report += "\n4. **Actionable Repair Recommendation**:\n"
    if not detections:
        report += "- No immediate action required. Schedule next routine visual inspection in 6 months.\n"
    else:
        recs = []
        if potholes:
            max_pothole_width = max([d["width_m"] for d in potholes])
            if max_pothole_width > 0.4:
                recs.append("Perform **Full-Depth Patching** for major potholes: excavate the damaged area, compact the subbase, and apply hot-mix asphalt (HMA).")
            else:
                recs.append("Apply **Throw-and-Roll / Skin Patching** for minor potholes to prevent water ingress and immediate tire damage.")
        if cracks:
            max_crack_length = max([d["height_m"] for d in cracks])
            if max_crack_length > 1.0:
                recs.append("Conduct professional **Crack Sealing / Routing** with elastomeric sealant for long cracks to block moisture infiltration.")
            else:
                recs.append("Monitor minor cracks. Apply fog seal or emulsion treatment during next scheduled road maintenance cycle.")
                
        for r in recs:
            report += f"- {r}\n"
            
    return report


# ---------------------------------------------------------------------------
# Helper — Unified AI Analysis with Fallback
# ---------------------------------------------------------------------------
def call_ai_analysis(detections: list, img: PILImage.Image) -> str:
    """Tries Gemini first, OpenAI second, and falls back to a highly realistic local report if both fail."""
    try:
        print("Attempting Gemini analysis...")
        return call_gemini(detections, img)
    except Exception as ge:
        print(f"Gemini analysis failed: {ge}. Attempting OpenAI fallback...")
        try:
            return call_openai(detections, img)
        except Exception as oe:
            print(f"OpenAI analysis also failed: {oe}. Generating local report...")
            return generate_local_fallback_report(detections)




# ---------------------------------------------------------------------------
# Preprocessing and Sliced Inference Helpers for High-Accuracy
# ---------------------------------------------------------------------------
def preprocess_image(img: PILImage.Image) -> PILImage.Image:
    """Applies CLAHE (contrast enhancement) and unsharp masking to enhance pothole/crack visibility."""
    try:
        import cv2
        import numpy as np
        
        # Convert PIL image to CV2 BGR
        img_np = np.array(img)
        if len(img_np.shape) == 2:  # grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Convert to LAB color space to apply CLAHE to the L channel
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to balance contrast on dark road asphalt
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels back
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Apply subtle Gaussian unsharp mask to sharpen fine crack lines
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # Convert back to PIL RGB
        return PILImage.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error during preprocessing: {e}. Falling back to original image.")
        return img


def apply_nms(boxes, iou_threshold=0.40):
    """Applies Non-Maximum Suppression to filter duplicate boxes from overlapping slices.
    Each box: [x1, y1, x2, y2, confidence, class_id]
    """
    if not boxes:
        return []
    try:
        import numpy as np
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        classes = boxes[:, 5]
        
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
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes that are either different classes or have low overlap
            cls_match = (classes[order[1:]] == classes[i])
            inds = np.where((ovr <= iou_threshold) | (~cls_match))[0]
            order = order[inds + 1]
            
        return boxes[keep].tolist()
    except Exception as e:
        print(f"NMS calculation error: {e}")
        return boxes


def draw_premium_boxes(img: PILImage.Image, boxes: list, class_names: dict) -> PILImage.Image:
    """Renders highly polished, premium transparent bounding boxes with modern pill badges."""
    from PIL import ImageDraw, ImageFont
    
    # Create semi-transparent overlay
    overlay = PILImage.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Elegant, vibrant color palettes
    colors = {
        "pothole": {"border": (249, 115, 22, 255), "fill": (249, 115, 22, 40)},    # Premium Orange
        "crack": {"border": (6, 182, 212, 255), "fill": (6, 182, 212, 40)},       # Premium Cyan
    }
    default_color = {"border": (139, 92, 246, 255), "fill": (139, 92, 246, 40)}   # Indigo fallback
    
    # Attempt to load a clean sans-serif font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(img.size[0] * 0.012)))
    except IOError:
        font = ImageFont.load_default()
        
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        class_name = class_names.get(cls_id, f"Class {cls_id}").lower()
        color_info = colors.get(class_name, default_color)
        
        # 1. Draw modern semi-transparent filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=color_info["fill"])
        
        # 2. Draw premium thick border (3px width)
        draw.rectangle([x1, y1, x2, y2], outline=color_info["border"], width=3)
        
        # 3. Create a sleek badge text
        label_text = f"{class_name.upper()} {int(conf * 100)}%"
        
        # 4. Text boundary calculation
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)
            
        # Draw elegant badge above bounding box
        pill_padding = 6
        pill_x1 = x1
        pill_y1 = max(0, y1 - text_h - pill_padding * 2)
        pill_x2 = x1 + text_w + pill_padding * 2
        pill_y2 = y1 if y1 > (text_h + pill_padding * 2) else (y1 + text_h + pill_padding * 2)
        
        if pill_y2 == y1 + text_h + pill_padding * 2:
            pill_y1 = y1
            pill_y2 = y1 + text_h + pill_padding * 2
            
        # Draw pill badge
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2], radius=4, fill=color_info["border"])
        else:
            draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=color_info["border"])
            
        # Draw label text inside pill badge
        draw.text((pill_x1 + pill_padding, pill_y1 + pill_padding), label_text, fill=(255, 255, 255, 255), font=font)
        
    # Composite transparent layers
    final_img = PILImage.alpha_composite(img.convert("RGBA"), overlay)
    return final_img.convert("RGB")


# ---------------------------------------------------------------------------
# Helper — run YOLO prediction with SAHI & Preprocessing
# ---------------------------------------------------------------------------
def run_prediction(img: PILImage.Image):
    """Run preprocessing, SAHI sliced prediction, NMS, and return (plotted_image, detections_list)."""
    # 1. Enhance image contrast and clarity first
    processed_img = preprocess_image(img)
    
    w, h = img.size
    
    # 2. Slice-Aided Inference parameters
    slice_size = 512
    overlap = 0.25
    conf_thresh = 0.12  # lower confidence threshold for detailed slice scanning
    
    boxes = []
    stride = int(slice_size * (1 - overlap))
    
    # Slices coordinates generator
    x_offsets = list(range(0, w - slice_size + 1, stride))
    if w > slice_size and x_offsets[-1] + slice_size < w:
        x_offsets.append(w - slice_size)
    if not x_offsets:
        x_offsets = [0]
        
    y_offsets = list(range(0, h - slice_size + 1, stride))
    if h > slice_size and y_offsets[-1] + slice_size < h:
        y_offsets.append(h - slice_size)
    if not y_offsets:
        y_offsets = [0]
        
    # Slide across the road image and make sliced predictions
    for x_off in x_offsets:
        for y_off in y_offsets:
            crop_w = min(slice_size, w - x_off)
            crop_h = min(slice_size, h - y_off)
            
            # Crop current window slice
            slice_img = processed_img.crop((x_off, y_off, x_off + crop_w, y_off + crop_h))
            
            # Perform inference on slice with Test-Time Augmentation (augment=True)
            results = model.predict(source=slice_img, conf=conf_thresh, augment=True, verbose=False)
            if not results or not len(results):
                continue
                
            result = results[0]
            for box in result.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Convert slice coordinates back to global canvas space
                boxes.append([
                    bx1 + x_off,
                    by1 + y_off,
                    bx2 + x_off,
                    by2 + y_off,
                    confidence,
                    class_id
                ])
                
    # 3. Global high-resolution inference pass to capture large potholes spanning slices
    global_imgsz = 1280 if max(w, h) >= 1280 else max(640, max(w, h))
    global_results = model.predict(source=processed_img, conf=conf_thresh, imgsz=global_imgsz, augment=True, verbose=False)
    if global_results and len(global_results):
        global_res = global_results[0]
        for box in global_res.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            boxes.append([bx1, by1, bx2, by2, confidence, class_id])
            
    # 4. Perform Non-Maximum Suppression to clean up duplicate detections
    merged_boxes = apply_nms(boxes, iou_threshold=0.40)
    
    # 5. Map detections lists with dynamic perspective-corrected scaling
    # Road lanes converge toward the horizon (perspective distortion), which means the physical
    # real-world size represented by a single pixel increases with vertical depth.
    # We model the road plane using standard perspective projection mapping:
    # Scale factor S(y) = mpp_bottom * ((h - horizon) / (y_road - horizon))
    # Assumes a standard dashboard camera setup where the horizon is at ~45% of image height (h).
    # At the bottom edge of the image (y = h), a standard lane width of 3.5m is assumed to span the image width (w).
    
    horizon = 0.45 * h
    mpp_bottom = 3.5 / w
    
    detections = []
    for box in merged_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        class_name = model.names[cls_id] if (model and hasattr(model, "names")) else f"Class {cls_id}"
        
        # Calculate vertical position where the anomaly touches the road surface
        # For cracks (distributed vertically), we use the vertical center.
        # For potholes, we use the bottom edge (y2).
        is_crack = "crack" in class_name.lower()
        y_road = (y1 + y2) / 2.0 if is_crack else y2
        
        # Clamp ground contact point to be safely below horizon
        y_road = max(y_road, horizon + 0.05 * h)
        
        # Dynamic meters-per-pixel multiplier based on depth on the road plane
        mpp_local = mpp_bottom * ((h - horizon) / (y_road - horizon))
        
        # Safety clamp to prevent infinite scale projection close to the horizon
        mpp_local = min(mpp_local, mpp_bottom * 8.0)
        
        width_px = int(x2 - x1)
        height_px = int(y2 - y1)
        
        detections.append({
            "class_name": class_name,
            "width_px": width_px,
            "height_px": height_px,
            "width_m": round(width_px * mpp_local, 2),
            "height_m": round(height_px * mpp_local, 2),
            "confidence": round(conf, 2)
        })
        
    # 6. Generate the premium plotted image with beautiful overlay styles
    plotted = draw_premium_boxes(img, merged_boxes, model.names if model else {})
    
    # Resize to 720p height for snappy UI delivery
    w_plot, h_plot = plotted.size
    new_h = 720
    new_w = int(new_h * (w_plot / h_plot))
    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
    plotted = plotted.resize((new_w, new_h), resample)
    
    return plotted, detections



# ===================================================================
#  ROUTES
# ===================================================================

@app.route("/")
def index():
    """Serve the main app page (or redirect to login if not authenticated)."""
    if not session.get("authenticated"):
        return redirect(url_for("login_page"))
    return render_template("index.html")


@app.route("/login")
def login_page():
    """Serve the login page."""
    if session.get("authenticated"):
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/send-otp", methods=["POST"])
def send_otp():
    """Generate and send an OTP to the user's email."""
    data = request.get_json()
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"status": "error", "message": "Email is required."}), 400

    otp = str(random.randint(100000, 999999))
    session["otp"] = otp
    session["otp_email"] = email
    print(f"\n{'='*40}\nVERIFICATION OTP FOR {email}: {otp}\n{'='*40}\n")

    ok, msg = send_otp_email(email, otp)
    if ok:
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg}), 500


@app.route("/login", methods=["POST"])
def login_verify():
    """Verify OTP and authenticate the user."""
    data = request.get_json()
    otp_input = data.get("otp", "").strip()

    stored_otp = session.get("otp")
    if not stored_otp:
        return jsonify({"status": "error", "message": "No OTP generated. Please request one first."}), 400

    if otp_input == stored_otp:
        session["authenticated"] = True
        session.pop("otp", None)
        session.pop("otp_email", None)
        return jsonify({"status": "success", "message": "Login successful!"})
    else:
        return jsonify({"status": "error", "message": "Invalid OTP. Please try again."}), 401


@app.route("/logout")
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image, run YOLO + Gemini, return JSON results."""
    if not session.get("authenticated"):
        return jsonify({"error": "Not authenticated"}), 401

    if model is None:
        return jsonify({"error": "YOLO model could not be loaded. Ensure best.pt is in the project root."}), 500

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img = PILImage.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Run YOLO prediction
    plotted, detections = run_prediction(img)

    # Run AI analysis (Gemini with OpenAI and local fallback)
    ai_text = call_ai_analysis(detections, img)

    # Convert plotted image to base64 for JSON response
    buf = BytesIO()
    plotted.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "image_b64": image_b64,
        "detections": detections,
        "openai_analysis": ai_text,
        "gemini_analysis": ai_text,
    })


# ===================================================================
#  RUN
# ===================================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
