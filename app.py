# ==============================
# app.py — Optimized for Raspberry Pi (Picamera2 + MTCNN)
# ==============================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, Response
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from datetime import datetime
from picamera2 import Picamera2
import cv2
import numpy as np

# ==============================
# 🔹 Flask Setup
# ==============================
app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# ==============================
# 🔹 Device & Model Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine

# ==============================
# 🔹 Load Model
# ==============================
model_path = r"/home/faceberry-pi/Desktop/face_recognition_app/model/ResNet18.pth"
train_dir = r"/home/faceberry-pi/Desktop/face_recognition_app/dataset"

class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

backbone = models.resnet18(pretrained=False)
backbone.fc = nn.Identity()
embedding_layer = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512))
model = nn.Sequential(backbone, embedding_layer).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['backbone_state_dict'])
model.eval()

# ==============================
# 🔹 Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# 🔹 MTCNN (slow but required)
# ==============================
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7])

# ==============================
# 🔹 Build Face Embedding Database
# ==============================
embedding_db = []
print("Building embedding database...")

with torch.no_grad():
    for class_name in class_names:
        folder = os.path.join(train_dir, class_name)
        embeddings = []
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            try:
                img = Image.open(path).convert("RGB")
                boxes, _ = mtcnn.detect(img)

                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                    face = img.crop((x1, y1, x2, y2))
                    tensor = transform(face).unsqueeze(0).to(device)

                    emb = model(tensor).cpu()
                    emb = F.normalize(emb, dim=1)
                    embeddings.append(emb)
            except Exception as e:
                print(f"Error: {img_name} — {e}")

        if embeddings:
            avg_emb = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=True)
            embedding_db.append((class_name, avg_emb))

print(f"✅ Database ready with {len(embedding_db)} identities.")

# ==============================
# 🔹 Match Function
# ==============================
def find_closest_match(face_emb, threshold=0.7):
    best_name, best_sim = "Unknown", -1
    for name, db_emb in embedding_db:
        sim = F.cosine_similarity(face_emb, db_emb).item()
        if sim > best_sim:
            best_sim, best_name = sim, name
    if best_sim < threshold:
        best_name = "Unknown"
    return best_name, best_sim

# ==============================
# 🔹 Flask Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    draw = ImageDraw.Draw(img)

    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face_crop = img.crop((x1, y1, x2, y2))
            tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                face_emb = model(tensor).cpu()
                face_emb = F.normalize(face_emb, dim=1)
            name, sim = find_closest_match(face_emb)
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
            draw.text((x1 + 5, y1 + 5), f"{name} ({sim:.2f})", fill='white')

    filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join("static", filename)
    img.save(path)
    return render_template("index.html", result_image=filename)

# ==============================
# 🔹 Live Streaming
# ==============================
def generate_frames():
    DETECT_W, DETECT_H = 320, 240  # downscaled for MTCNN

    while True:
        frame = picam2.capture_array()  # 640×480 RGB
        frame_small = cv2.resize(frame, (DETECT_W, DETECT_H))
        img_small = Image.fromarray(frame_small)

        boxes, _ = mtcnn.detect(img_small)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if boxes is not None:
            scale_x = frame.shape[1] / DETECT_W
            scale_y = frame.shape[0] / DETECT_H

            for box in boxes:
                x1s, y1s, x2s, y2s = box
                x1, y1 = int(x1s * scale_x), int(y1s * scale_y)
                x2, y2 = int(x2s * scale_x), int(y2s * scale_y)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                pil_face = Image.fromarray(face_crop).convert("RGB")
                tensor = transform(pil_face).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(tensor).cpu()
                    emb = F.normalize(emb, dim=1)

                name, sim = find_closest_match(emb)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"{name} ({sim:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================
# 🔹 Run App
# ==============================
if __name__ == '__main__':
    picam2 = None
    # Initialize Picamera2 only in Flask reloader child process
    if "WERKZEUG_RUN_MAIN" in os.environ:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print("📷 Picamera2 initialized.")

    app.run(host='0.0.0.0', port=5000, debug=True)
