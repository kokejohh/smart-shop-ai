import cv2
import numpy as np
from insightface.app import FaceAnalysis

# --- SETUP: โมเดล ---
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- SETUP: โหลดรูปต้นฉบับ ---
print("Loading reference face...")
img_ref = cv2.imread("me.jpg")
if img_ref is None:
    print("Error: หาไฟล์ me.jpg ไม่เจอ")
    exit()

faces_ref = app.get(img_ref)
if len(faces_ref) == 0:
    print("Error: ไม่พบใบหน้าในรูป me.jpg")
    exit()

ref_embedding = faces_ref[0].normed_embedding
ref_name = "Koke"

# ฟังก์ชันคำนวณความเหมือน
def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2)

# --- Runtime Setup ---
cap = cv2.VideoCapture(0)

# [Tuning] กำหนดว่าจะให้คำนวณใหม่ทุกๆ กี่เฟรม
# ค่า 5 = คำนวณ 1 ครั้ง พัก 4 ครั้ง (ถ้าคอมช้าให้เพิ่มเป็น 10-15)
SKIP_FRAMES = 10 
frame_count = 0

# ตัวแปรสำหรับจำค่าหน้าคน เพื่อเอามาวาดในเฟรมที่ไม่ได้คำนวณ
saved_faces_to_draw = []

print(f"Start Recognition... Processing every {SKIP_FRAMES} frames.")
print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret: break

    # เพิ่มตัวนับเฟรม
    frame_count += 1

    # --- LOGIC: ตรวจสอบเฉพาะรอบที่กำหนด (Method 2 Concept) ---
    if frame_count % SKIP_FRAMES == 0:
        
        # 1. สั่ง AI คำนวณ (เฉพาะจังหวะนี้ที่กินแรง CPU)
        faces = app.get(frame)
        
        # 2. เคลียร์ค่าเก่า เตรียมเก็บค่าใหม่
        saved_faces_to_draw = []

        for face in faces:
            # คำนวณความเหมือน
            current_embedding = face.normed_embedding
            sim = compute_sim(ref_embedding, current_embedding)
            
            # Logic การตัดสินใจ
            if sim > 0.5:
                text = f"{ref_name} ({sim:.2f})"
                color = (0, 255, 0) # เขียว
            else:
                text = f"Unknown ({sim:.2f})"
                color = (0, 0, 255) # แดง
            
            # แปลง Bbox และ Keypoints เป็น int ให้พร้อมวาด
            bbox = face.bbox.astype(int)
            kps = face.kps.astype(int)

            # เก็บข้อมูลลง Cache ไว้ใช้วาดในเฟรมถัดๆ ไป
            saved_faces_to_draw.append({
                'bbox': bbox,
                'text': text,
                'color': color,
                'kps': kps
            })
            
    # --- DRAWING: วาดรูปทุกเฟรม (โดยใช้ข้อมูลจาก Cache) ---
    # ส่วนนี้ทำงานเร็วมาก เพราะแค่ตีกรอบสี่เหลี่ยม ไม่มีการคำนวณ AI
    for item in saved_faces_to_draw:
        bbox = item['bbox']
        color = item['color']
        text = item['text']
        kps = item['kps']

        # วาดกรอบ
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # วาดชื่อ
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # วาดจุดบนหน้า
        for kp in kps:
            cv2.circle(frame, (kp[0], kp[1]), 2, (255, 0, 0), -1)

    cv2.imshow('InsightFace Optimized', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
