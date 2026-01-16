import cv2
import math
import numpy as np
import mediapipe as mp
import asyncio

from insightface.app import FaceAnalysis
from qdrant_client import AsyncQdrantClient, QdrantClient, models

next_id = 0
tracked_persons = {}
MAX_LOST_FRAMES = 10

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_thumb_up(hand_landmarks):
    is_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y < hand_landmarks.landmark[2].y
 
    fingers_folded = (
        hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and
        hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
        hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
        hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    )
    
    return is_up and fingers_folded

async def main():
    global next_id, tracked_persons

    
    qdrant_client = AsyncQdrantClient(
        url="https://c3965568-2dcf-4d83-9e90-4361ca25d011.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DSSmGBHT9bi_XI9T8a9J-dBgH9trfnPydKkmWB2pT0M",
    )

    local_cache = QdrantClient(':memory:')
    local_cache.create_collection(
        collection_name='face_cache',
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )

    print(f"!!!{qdrant_client.get_collections()}")

    appFace = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider','CPUExecutionProvider'])

    appFace.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    box_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    keypoint_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)

    SKIP_FRAMES = 1
    frame_count = 0

    saved_faces_to_draw = []

    print(f"Start Recognition... Processing every {SKIP_FRAMES} frames.")
    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_count += 1

        faces = appFace.get(frame)
        saved_faces_to_draw = []

        current_faces_this_frame = []

        for face in faces:
            current_embedding = face.normed_embedding
          
            bbox = face.bbox.astype(int)
            kps = face.kps.astype(int)

            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            found_id = None
            min_dist = 150

            for p_id, data in tracked_persons.items():
                dist = get_distance(center, data['center'])
                if dist < min_dist:
                    found_id = p_id
                    min_dist = dist
                    break

            if found_id is None:
                found_id = next_id

                tracked_persons[found_id] = {
                    'center': center,
                    'name': 'Scanning...',
                    'lost_frames': 0
                }

                next_id += 1

                local_results = local_cache.query_points(
                    collection_name='face_cache',
                    query=current_embedding,
                    limit=1,
                    score_threshold=0.5
                ).points

                face_results = local_results

                if not local_results:
                    cloud_results = (await qdrant_client.query_points(
                        collection_name="face",
                        query=current_embedding.tolist(),
                        limit=1,
                        score_threshold=0.5
                    )).points

                    if cloud_results:
                        res = cloud_results[0]

                        local_cache.upsert(
                            collection_name='face_cache',
                            points=[
                                models.PointStruct(
                                    id=res.id,
                                    vector=res.vector if res.vector else current_embedding.tolist(),
                                    payload=res.payload
                                )
                            ]
                        )
                    face_results = cloud_results
                tracked_persons[found_id]['name'] = face_results[0].payload.get('name', 'Unknown') if face_results else 'Unknown'
                tracked_persons[found_id]['score'] = face_results[0].score if face_results else 0
                if tracked_persons[found_id]['name'] == 'Unknown':
                    tracked_persons[found_id]['retry'] = 0
            else:
                tracked_persons[found_id]['center'] = center
                tracked_persons[found_id]['lost_frames'] = 0
            
            if tracked_persons[found_id]['name'] == 'Unknown' and tracked_persons[found_id]['retry'] < 3:
                tracked_persons[found_id]['retry'] += 1
                local_results = local_cache.query_points(
                    collection_name='face_cache',
                    query=current_embedding,
                    limit=1,
                    score_threshold=0.5
                ).points

                face_results = local_results

                if tracked_persons[found_id]['retry'] == 2:
                    if not local_results:
                        cloud_results = (await qdrant_client.query_points(
                            collection_name="face",
                            query=current_embedding.tolist(),
                            limit=1,
                            score_threshold=0.5
                        )).points

                        if cloud_results:
                            res = cloud_results[0]

                            local_cache.upsert(
                                collection_name='face_cache',
                                points=[
                                    models.PointStruct(
                                        id=res.id,
                                        vector=res.vector if res.vector else current_embedding.tolist(),
                                        payload=res.payload
                                    )
                                ]
                            )
                        face_results = cloud_results
                tracked_persons[found_id]['name'] = face_results[0].payload.get('name', 'Unknown') if face_results else 'Unknown'
                tracked_persons[found_id]['score'] = face_results[0].score if face_results else 0

            tracked_persons[found_id]['bbox'] = bbox
            tracked_persons[found_id]['kps'] = kps
            if tracked_persons[found_id]['name'] == 'Unknown':
                tracked_persons[found_id]['color'] = color = (0, 0, 255)
            else:
                tracked_persons[found_id]['color'] = color = (0, 255, 0)

            current_faces_this_frame.append(found_id)

        for p_id in list(tracked_persons.keys()):
            data = tracked_persons[p_id]
            bbox = data['bbox']
            color = data['color']
            kps = data['kps']

            cv2.putText(frame, f"{data['name']}: ({data['score']:.2f})", (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            for kp in kps:
                cv2.circle(frame, (kp[0], kp[1]), 2, (255, 0, 0), -1)
        
            if p_id not in current_faces_this_frame:
                data['lost_frames'] += 1
                if data['lost_frames'] > MAX_LOST_FRAMES:
                    del tracked_persons[p_id]

        result_hands = hands.process(rgb_frame)
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
             
        cv2.imshow('Smart Shop', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())