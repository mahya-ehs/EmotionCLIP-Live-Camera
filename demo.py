import cv2
import clip
import torch
import numpy as np
import threading
from PIL import Image
from ultralytics import YOLO
from src.models.base import EmotionCLIP
from torchvision.transforms import functional as F
from torchvision import transforms

# Define classifier classes, write any classifier you want
# We made some predefined classifiers you can uncomment

# Emotions
PREDICTION_CLASSES = ['neutral', 'happy', 'angry', 'sad', 'disgust']

# Sentiment
# PREDICTION_CLASSES = ['positive', 'negative', 'neutral']

# Alertness
# PREDICTION_CLASSES = ['eyes open', 'eyes closed']

# Emotions with man, woman or child
# PREDICTION_CLASSES = ['neutral man', 'neutral woman', 'neutral child', 'happy man', 'happy woman', 'happy child', 'sad man', 'sad woman', 'sad child', 'angry man', 'angry woman', 'angry child']

# Age
# PREDICTION_CLASSES = ['0 year old', '10 year old', '20 year old', '30 year old', '40 year old', '50 year old', '60 year old', '70 year old', '80 year old', '90 year old', '100 year old']

# Actions
# PREDICTION_CLASSES = ['drinking', 'eating', 'sleeping', 'neutral', 'sitting', 'standing']

# Objects
# PREDICTION_CLASSES = ['cat', 'dog', 'book', 'bottle', 'chair']

# Emotions and actions and objects (conflicting classes, for testing only, still does well tho)
# PREDICTION_CLASSES = ['neutral', 'happy', 'angry', 'sad', 'disgust', 'drinking', 'eating', 'sleeping', 'neutral', 'cat', 'dog', 'book', 'bottle', 'chair']


# Global constants
YOLO_PERSON_CLASS_ID = 0
ORIGINAL_FRAME_HEIGHT = 480
ORIGINAL_FRAME_WIDTH = 640
PREDICTIONS_EVERY_N_FRAME = 40
# Use absolute path for now, because we are sending the path to another file in a different folder
BACKBONE_CHECKPOINT_ABSPATH = "C:/Users/Krist/Sisu_EmotionCLIP-Live-Camera/exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt"


# Global variables
# label and person boxes needs to be global, so we can draw them on the frame when multithreading
label = ""
person_boxes = []


# Load models
device = torch.device("cpu")
yolo_model = YOLO("yolo11n.pt")
emotionclip_model = EmotionCLIP(backbone_checkpoint=BACKBONE_CHECKPOINT_ABSPATH)
ckpt = torch.load(BACKBONE_CHECKPOINT_ABSPATH, map_location='cpu')
emotionclip_model.load_state_dict(ckpt['model'], strict=True)
emotionclip_model.eval()


# Processed global constant
PREDICTION_CLASSES_TOKENIZED = clip.tokenize(PREDICTION_CLASSES).to(device)


def bbox_to_mask(bbox, target_shape):
    mask = torch.zeros(target_shape[1], target_shape[0])

    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    return mask


def preprocess_frames(
        frames: list[Image.Image],
        bboxes: list[list[float]],
        crop_method: str = 'center',
) -> tuple[torch.Tensor, torch.Tensor]:
    RESIZE_SIZE = 256
    CROP_SIZE = 224
    processed_frames = []
    processed_masks = []
    for frame, bbox in zip(frames, bboxes):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        print(bbox)
        mask = bbox_to_mask(bbox, (480,640))
        resized_frame = F.resize(pil_image, size=RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)
        resized_mask = F.resize(mask.unsqueeze(0), size=RESIZE_SIZE).squeeze()
        if crop_method == 'center':
            cropped_frame = F.center_crop(resized_frame, output_size=CROP_SIZE)
            cropped_mask = F.center_crop(resized_mask, output_size=CROP_SIZE)
        elif crop_method == 'random':
            i, j, h, w = transforms.RandomCrop.get_params(resized_frame, output_size=CROP_SIZE)
            cropped_frame = F.crop(resized_frame, i, j, h, w)
            cropped_mask = F.crop(resized_mask, i, j, h, w)
        else:
            raise NotImplementedError
        normalized_frame = F.normalize(
            tensor=F.to_tensor(cropped_frame),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        binarized_mask = (cropped_mask > 0.5).long()
        processed_frames.append(normalized_frame)
        processed_masks.append(binarized_mask)

    video = torch.stack(processed_frames, dim=0).float()
    video_mask = torch.stack(processed_masks, dim=0).long()
    return video, video_mask



def predict_emotion(frame):
    global person_boxes
    yolo_results = yolo_model(frame)

    # Extract person bounding boxes
    if yolo_results:
        for r in yolo_results:
            if r.boxes:  # Ensure that there are bboxes in the result
                boxes = r.boxes.xyxy  # Get the bboxes in the right format
                classes = r.boxes.cls  # Get the class predictions for each bbox

                # Filter boxes that correspond to the "person" class
                person_boxes = []
                for box, cls in zip(boxes, classes):
                    if cls == YOLO_PERSON_CLASS_ID:
                        person_boxes.append(box.tolist())

    # Preprocess
    frame_preprocessed, mask_preprocessed = preprocess_frames([frame], person_boxes)

    # Encode frame and labels
    with torch.no_grad():
        image_features = emotionclip_model.encode_image(frame_preprocessed, mask_preprocessed)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        emotion_features = emotionclip_model.encode_text(PREDICTION_CLASSES_TOKENIZED)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)

    # Get predictions with logits
    logits_per_image = (100.0 * image_features @ emotion_features.T).softmax(dim=-1)     # size: (B, class_num)

    probs = logits_per_image.cpu().numpy()[0]
    for i, emotion in enumerate(PREDICTION_CLASSES):
        print(f"Probability of {emotion}: {probs[i]*100:.2f}%")
    # Using global because of multithreading
    global label
    max_prob_idx = np.argmax(probs)
    max_emotion = PREDICTION_CLASSES[max_prob_idx]
    max_prob = f"{probs[max_prob_idx]*100:.2f}%"
    print(f"Prediction: {max_emotion}\n")
    label = f"{max_emotion}, {max_prob}"


def draw_label(frame, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color
    thickness = 2
    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    # Set the position of the label, not bound to bbox as the emotion prediction is done on the whole frame
    label_x = 0
    label_y = 20

    # Draw the label background
    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), (label_x + label_size[0], label_y + 5), (0, 255, 0),
                  cv2.FILLED)

    # Put the label on top of the rectangle
    cv2.putText(frame, label, (label_x, label_y), font, font_scale, font_color, thickness)


def draw_bboxes(frame, person_boxes):
    for box in person_boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)


def main():
    # opening webcam
    cam = cv2.VideoCapture(0)
    frame_count = 0
    processing_thread = None
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Only run every N frames
        if frame_count % PREDICTIONS_EVERY_N_FRAME == 0:
            # If there is no ongoing thread, start a new one to process the frame
            if processing_thread is None or not processing_thread.is_alive():
                processing_thread = threading.Thread(target=predict_emotion, args=(frame,))
                processing_thread.start()

        draw_label(frame, label)
        draw_bboxes(frame, person_boxes)

        frame = cv2.resize(frame, (ORIGINAL_FRAME_WIDTH*2, ORIGINAL_FRAME_HEIGHT*2))
        cv2.imshow('Webcam, prediction, bounding boxes', frame)

        frame_count += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
