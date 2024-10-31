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

# Global constants
EMOTION_CLASSES =[
    'happy',
    'sad',
    'angry',
    'sleeping',
    'cat'
]
YOLO_PERSON_CLASS_ID = 0
ORIGINAL_FRAME_HEIGHT = 480
ORIGINAL_FRAME_WIDTH = 640
ENCODING_FRAME_HEIGHT = 224
ENCODING_FRAME_WIDTH = 224
PREDICTIONS_EVERY_N_FRAME = 60
BACKBONE_CHECKPOINT_ABSPATH = "C:/Users/Krist/EmotionCLIP-Assignment/exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt"

# Global variables
# label and person boxes needs to be global, so we can draw them on the frame when multithreading and limiting fps
label = ""
person_boxes = []

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolo11n.pt")
emotionclip_model = EmotionCLIP(backbone_checkpoint=BACKBONE_CHECKPOINT_ABSPATH)
ckpt = torch.load(BACKBONE_CHECKPOINT_ABSPATH, map_location='cpu')
emotionclip_model.load_state_dict(ckpt['model'], strict=True)
emotionclip_model.eval()

# Processed global constant
EMOTION_CLASSES_TOKENIZED = clip.tokenize(EMOTION_CLASSES).to(device)


# Helper function to create an empty mask
def create_empty_mask(height, width):
    return torch.zeros((height, width), dtype=torch.uint8)


# Processed global variable
# Combined mask is global just so that we can debug the mask visually with cv2.imshow()
combined_mask = create_empty_mask(ORIGINAL_FRAME_HEIGHT, ORIGINAL_FRAME_WIDTH)


# Convert bounding box to mask
# Function stolen from base.py, and slightly altered to fit our bbox structure
# Note that they put the mask value to 1 instead of 255, I don't know if this is the correct way to do it
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
    global combined_mask
    global person_boxes

    yolo_results = yolo_model(frame)
    person_class_id = 0

    # View results
    if yolo_results:
        for r in yolo_results:
            if r.boxes:  # Ensure that there are bboxes in the result
                boxes = r.boxes.xyxy  # Get the bboxes in the right format
                classes = r.boxes.cls  # Get the class predictions for each bbox

                # Filter boxes that correspond to the "person" class
                person_boxes = []
                for box, cls in zip(boxes, classes):
                    if cls == person_class_id:
                        person_boxes.append(box.tolist())

                person_masks = []
                for box in person_boxes:
                    person_masks.append(bbox_to_mask(box, (ORIGINAL_FRAME_WIDTH, ORIGINAL_FRAME_HEIGHT)))  # The given function has height and width swapped for reasons I don't understand

                # Emptying combined mask as late as possible because this is being multithreaded
                combined_mask = create_empty_mask(ORIGINAL_FRAME_HEIGHT, ORIGINAL_FRAME_WIDTH)
                for mask in person_masks:
                    # Combines all masks
                    # This forces the mask to be 0's and 1's, we might want to replace 1 with 255, but I don't know
                    combined_mask = torch.logical_or(combined_mask, mask).to(torch.uint8)

                    # This forces it to 255, but it doesn't seem to make a difference
                    # combined_mask[combined_mask == 1] = 255
            else:
                # Emptying combined mask as late as possible because this is being multithreaded
                # Create an empty mask filled with 0s
                combined_mask = create_empty_mask(ORIGINAL_FRAME_HEIGHT, ORIGINAL_FRAME_WIDTH)
    else:
        # Emptying combined mask as late as possible because this is being multithreaded
        # Create an empty mask filled with 0s
        combined_mask = create_empty_mask(ORIGINAL_FRAME_HEIGHT, ORIGINAL_FRAME_WIDTH)

    frame_test, mask_test = preprocess_frames([frame], person_boxes)
    frame_inner = cv2.resize(frame, (ENCODING_FRAME_HEIGHT, ENCODING_FRAME_WIDTH))
    combined_mask_numpy = cv2.resize(combined_mask.numpy(), (ENCODING_FRAME_HEIGHT, ENCODING_FRAME_WIDTH))
    frame_tensor = torch.from_numpy(frame_inner.transpose(2, 0, 1)).float()  # Convert to float32 and normalize
    mask_tensor = torch.from_numpy(combined_mask_numpy)
    # image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    C, H, W = frame_inner.transpose(2, 0, 1).shape
    mask_tensor = mask_tensor.unsqueeze(0)
    frame_tensor = frame_tensor.reshape(-1, C, H, W)
    mask_tensor = mask_tensor.reshape(-1, H, W)

    with torch.no_grad():
        image_features = emotionclip_model.encode_image(frame_test, mask_test)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        emotion_features = emotionclip_model.encode_text(EMOTION_CLASSES_TOKENIZED)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)


    logits_per_image = (100.0 * image_features @ emotion_features.T).softmax(dim=-1)     # size: (B, class_num)

    probs = logits_per_image.cpu().numpy()[0]
    for i, emotion in enumerate(EMOTION_CLASSES):
        print(f"Probability of {emotion}: {probs[i]*100:.2f}%")
    # Using global because of multithreading
    global label
    max_prob_idx = np.argmax(probs)
    max_emotion = EMOTION_CLASSES[max_prob_idx]
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
        cv2.imshow('Webcam, prediction, bounding boxes', frame)
        cv2.imshow('Combined mask', combined_mask.numpy())

        frame_count += 1

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
