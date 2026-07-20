import argparse
import collections
import cv2
import json
import logging
import numpy as np
import time
from pathlib import Path

import torch

from model_3dcnn import Improved3DCNN
from preprocessing import ImprovedVideoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    idx_to_class = checkpoint.get("class_mapping")
    if not idx_to_class:
        logger.error("No class mapping in checkpoint!")
        return
        
    num_classes = len(idx_to_class)
    model = Improved3DCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Recognizing {num_classes} classes.")

    # Initialize processor (same exact settings as training)
    processor = ImprovedVideoProcessor(
        target_size=(112, 112),
        num_frames=30,
        use_hand_detection=True,
        confidence=0.3
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return

    # Frame buffer to keep the last 30 frames
    frame_buffer = collections.deque(maxlen=30)
    
    # State for predictions
    current_prediction = "Waiting for frames..."
    current_confidence = 0.0
    last_inference_time = time.time()
    
    logger.info("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from camera. Exiting.")
            break
            
        frame = cv2.flip(frame, 1)  # Mirror view
        
        # We need to run the preprocessor's frame-level logic on the raw frame
        # which involves HandDetection + ROI + CLAHE + Resize + Normalize.
        try:
            processed_frame = processor._process_frame(frame)
            frame_buffer.append(processed_frame)
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            pass
            
        # Draw bounding box for visualization if hand detected
        vis_frame = frame.copy()
        if processor.detector:
            bbox = processor.detector.find_hand_bbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(vis_frame, "Hand Detected", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Run inference if buffer is full and enough time has passed
        now = time.time()
        if len(frame_buffer) == 30 and (now - last_inference_time) > args.inference_interval:
            # Prepare tensor: (C, T, H, W)
            frames_np = np.array(frame_buffer) # (30, 112, 112, 3)
            tensor = torch.from_numpy(frames_np).float()
            tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0) # (1, 3, 30, 112, 112)
            tensor = tensor.to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                conf = conf.item()
                if conf > args.threshold:
                    current_prediction = idx_to_class[str(pred_idx.item())]
                    current_confidence = conf
                else:
                    current_prediction = "Uncertain..."
                    current_confidence = conf
                    
            last_inference_time = now

        # Display results
        cv2.putText(vis_frame, f"Pred: {current_prediction}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vis_frame, f"Conf: {current_confidence:.2f}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        buffer_fill = len(frame_buffer) / 30.0
        cv2.rectangle(vis_frame, (10, 100), (int(10 + buffer_fill * 200), 120), (0, 255, 255), -1)
        cv2.putText(vis_frame, "Buffer", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.imshow("Sign Language Recognition", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition via Webcam")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--inference_interval", type=float, default=0.5, help="Seconds between inferences")
    
    args = parser.parse_args()
    main(args)
