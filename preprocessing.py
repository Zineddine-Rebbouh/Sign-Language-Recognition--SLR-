import cv2
import numpy as np
import logging

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class HandDetector:
    """MediaPipe hands detector for ROI extraction."""
    
    def __init__(self, confidence: float = 0.3):
        if mp is None:
            raise ImportError("MediaPipe is required. Run: pip install mediapipe")
            
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )

    def find_hand_bbox(self, frame_rgb: np.ndarray, padding: float = 0.2) -> tuple:
        """Find bounding box around detected hands.
        
        Args:
            frame_rgb: RGB frame.
            padding: Margin to add around the detected hand, as a fraction of its size.
            
        Returns:
            (x_min, y_min, x_max, y_max) or None if no hand detected.
        """
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None

        h, w, _ = frame_rgb.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

        # Add padding
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)

        x_min = max(0, x_min - pad_w)
        y_min = max(0, y_min - pad_h)
        x_max = min(w, x_max + pad_w)
        y_max = min(h, y_max + pad_h)

        if x_max <= x_min or y_max <= y_min:
            return None

        return (x_min, y_min, x_max, y_max)


class ImprovedVideoProcessor:
    """Video processing pipeline: Hand detection -> ROI crop -> CLAHE -> Resize."""
    
    def __init__(
        self,
        target_size: tuple = (112, 112),
        num_frames: int = 30,
        use_hand_detection: bool = True,
        confidence: float = 0.3
    ):
        self.target_size = target_size
        self.num_frames = num_frames
        self.use_hand_detection = use_hand_detection
        
        self.detector = HandDetector(confidence=confidence) if use_hand_detection else None
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        bbox = None
        if self.use_hand_detection and self.detector:
            bbox = self.detector.find_hand_bbox(rgb)
            
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            roi = frame[y_min:y_max, x_min:x_max]
        else:
            roi = frame  # Fallback to full frame

        # Convert ROI to LAB color space for CLAHE
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Resize
        resized = cv2.resize(enhanced, self.target_size)
        
        # Convert to RGB and normalize to [0, 1] float32
        final = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return final.astype(np.float32) / 255.0

    def process_video(self, video_path: str) -> np.ndarray:
        """Process a video file into a standard tensor format.
        
        Returns:
            frames: Array of shape (num_frames, H, W, C)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(self._process_frame(frame))
        finally:
            cap.release()
            
        if not frames:
            # Return zero tensor if video is unreadable
            return np.zeros((self.num_frames, *self.target_size, 3), dtype=np.float32)
            
        frames = np.array(frames)
        total_frames = len(frames)
        
        # Temporal padding/truncation to exactly `num_frames`
        if total_frames > self.num_frames:
            # Uniformly sample frames
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = frames[indices]
        elif total_frames < self.num_frames:
            # Pad by repeating the last frame
            pad_len = self.num_frames - total_frames
            pad_frames = np.repeat(frames[-1:], pad_len, axis=0)
            frames = np.concatenate([frames, pad_frames], axis=0)
            
        return frames
