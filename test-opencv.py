import cv2
import numpy as np
from src.models.base import EmotionCLIP
import torch

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

# Load the pre-trained model
model = EmotionCLIP(backbone_checkpoint=None)
ckpt = torch.load("emotionclip_latest.pt", map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)
model.eval()

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    mask = np.zeros_like(gray_image)  # Create a blank mask (same size as the frame)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Draw bounding box
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)  # Fill mask with white on detected faces

    return faces, mask

while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    # shape of visual mask and frames in linear_eval : (128, 8, 3, 224, 224)

    faces, face_mask = detect_bounding_box(video_frame)  # detect faces and generate mask

    # Resize frame and mask to match the input size expected by the model
    frame = cv2.resize(video_frame, (224, 224))
    mask = cv2.resize(face_mask, (224, 224))
    
    frame = frame.transpose(2, 0, 1)

    # Convert to torch tensors
    frame_tensor = torch.from_numpy(frame).float() / 255.0  # Convert to float32 and normalize
    mask_tensor = torch.from_numpy(mask).float() / 255.0  # Convert to float32 and normalize
    
    # Transpose frame and mask size to (3, 224, 224) and (1, 224, 224)
    mask_tensor = mask_tensor.unsqueeze(0)
    
    # # Add batch dimension: (1, 3, 224, 224)
    # video_frame_tensor = video_frame_tensor.unsqueeze(0)
    # face_mask_tensor = face_mask_tensor.unsqueeze(0)

    frame_tensor = frame_tensor.repeat(1, 8, 1, 1, 1)
    mask_tensor = mask_tensor.repeat(1, 8, 1, 1, 1)
    
    features = model.encode_video(frame_tensor, mask_tensor)

    with torch.no_grad():
        output = model(features)

    print(output)    
    cv2.imshow("Face Detection", video_frame)  # display the frame with bounding boxes
    cv2.imshow("Face Mask", face_mask)  # display the mask of the faces

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
