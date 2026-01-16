import cv2
import uuid
import numpy as np


from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

qdrant_client = QdrantClient(
    url="https://c3965568-2dcf-4d83-9e90-4361ca25d011.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DSSmGBHT9bi_XI9T8a9J-dBgH9trfnPydKkmWB2pT0M",
)

print(f"!!!!{qdrant_client.get_collections()}")

appFace = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
appFace.prepare(ctx_id=0, det_size=(640, 640))

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2)

def register_face(image_path, person_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Not found image")
        return
    faces = appFace.get(img)
    if len(faces) == 0:
        print(f"Not found face")
        return
    target_face = faces[0]
    embedding = target_face.embedding.tolist()
    
    point_id = str(uuid.uuid4())

    operation_info = qdrant_client.upsert(
        collection_name = 'face',
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "name": person_name
                }
            )
        ]
    )
    print(f"record '{person_name}' status: {operation_info.status}")

if __name__ == '__main__':
    register_face('pat.jpg', 'pattyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
