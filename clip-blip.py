import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import clip
import torch
import numpy as np

# CLIP for detecting emotion (for giving the prob of each class)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

emotion_classes = ["happy", "sad", "angry", "neutral", "sleepy", "cat"]
emotion_texts = clip.tokenize(emotion_classes).to(device)

# BLIP for generating caption per frame
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

def detect_emotion(frame, caption):
    image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    caption_input = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        caption_features = clip_model.encode_text(caption_input)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    caption_features /= caption_features.norm(dim=-1, keepdim=True)

    combined_features = (image_features + caption_features) / 2

    with torch.no_grad():
        emotion_features = clip_model.encode_text(emotion_texts)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)

    logits_per_image = (100.0 * combined_features @ emotion_features.T).softmax(dim=-1)     # size: (B, class_num)

    probs = logits_per_image.cpu().numpy()[0]
    for i, emotion in enumerate(emotion_classes):
        print(f"Probability of {emotion}: {probs[i]*100:.2f}%")
    max_prob_idx = np.argmax(probs)
    max_emotion = emotion_classes[max_prob_idx]

# opening webcam
cap = cv2.VideoCapture(0)  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    caption = generate_caption(frame)
    detect_emotion(frame, caption)

cap.release()
cv2.destroyAllWindows()
