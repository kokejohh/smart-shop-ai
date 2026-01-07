import cv2
import threading
import time
import mediapipe as mp

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

output_frame: list[bytes | None] = [None, None]
lock = threading.Lock()

@app.get("/")
def read_root():
    return StreamingResponse(generate_frame(0), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/1")
def read_root2():
    return StreamingResponse(generate_frame(1), media_type="multipart/x-mixed-replace; boundary=frame")

def camera_stream(index, camera_index):
    global output_frame

    cap = cv2.VideoCapture(camera_index)

    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    

    box_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    keypoint_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)

    while True:
        success, frame = cap.read()

        if not success:
            time.sleep(0.1)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection,
                                          bbox_drawing_spec=box_spec,
                                keypoint_drawing_spec=keypoint_spec)
                h, w, _ = frame.shape
                box = detection.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)

                cv2.putText(frame, "Detect", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        result2 = hands.process(rgb_frame)
        if result2.multi_hand_landmarks:
            for hand_landmarks in result2.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )


        ret, buffer = cv2.imencode('.jpg', frame)
        output_frame[index] = buffer.tobytes()
        if not ret:
            with lock:
                output_frame[index] = buffer.tobytes()
        time.sleep(0.1)


def generate_frame(index):
    global output_frame
    while True:
        with lock:
            if output_frame[index] is None:
                current_bytes = None
            else:
                current_bytes = output_frame[index]
        if current_bytes is None:
            time.sleep(0.01)
            continue
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + current_bytes + b'\r\n')
        time.sleep(0.03)


t1 = threading.Thread(target=camera_stream, args=(0, 0))
t1.daemon = True
t1.start()


#t2 = threading.Thread(target=camera_stream, args=(1, 'http://192.168.1.4:8080/video',))
#t2.daemon = True
#t2.start()
