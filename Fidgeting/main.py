import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from enum import Enum


# Define hand states
class HandState(Enum):
    NORMAL = 0
    FIDGETING = 1
    PINCHING = 2
    TAPPING = 3  # New state for finger tapping


# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# Simple exponential filter for smoothing
class ExpFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def __call__(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# Main hand fidget detector class
class HandFidgetDetector:
    def __init__(self):
        # Configuration parameters
        self.WINDOW_LEN = 20
        self.FIDGET_THRESHOLD = 5
        self.PINCH_DISTANCE_THRESHOLD = 30
        self.COOLDOWN_TIME = 30
        self.DISPLAY_TIME = 60

        # Initialize MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Initialize trackable landmarks
        self.tip_ids = [
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]

        self.pinch_fingers = [
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        # Initialize filters and data structures
        self.init_filters()
        self.hands_data = {'Left': self.create_hand_data(), 'Right': self.create_hand_data()}

        # For visualization
        self.frame_counter = 0

    def init_filters(self):
        """Initialize smoothing filters for all landmarks"""
        self.filters = {
            'Left': [[ExpFilter() for _ in range(21)], [ExpFilter() for _ in range(21)]],
            'Right': [[ExpFilter() for _ in range(21)], [ExpFilter() for _ in range(21)]]
        }

    def create_hand_data(self):
        """Create data structure for tracking a hand"""
        return {
            'pos_history': {t: deque(maxlen=self.WINDOW_LEN) for t in self.tip_ids},
            'vel_history': {t: deque(maxlen=self.WINDOW_LEN) for t in self.tip_ids},
            'dir_changes': {t: deque(maxlen=self.WINDOW_LEN) for t in self.tip_ids},
            'pinch_dist': deque(maxlen=self.WINDOW_LEN),
            'pinch_events': deque(maxlen=self.WINDOW_LEN),
            'tap_events': deque(maxlen=self.WINDOW_LEN),
            'last_pinch': False,
            'cooldown': 0,
            'display_counter': 0,
            'state': HandState.NORMAL,
            'avg_speed': 0
        }

    def setup_ui(self):
        """Create UI controls"""
        cv2.namedWindow("Controls")
        cv2.createTrackbar("Speed Thresh", "Controls", 5, 20, lambda x: None)
        cv2.createTrackbar("Dir Changes", "Controls", 7, 20, lambda x: None)
        cv2.createTrackbar("Pinch Dist", "Controls", 30, 100, lambda x: None)

    def update_params(self):
        """Get parameters from trackbars"""
        return {
            'speed_threshold': cv2.getTrackbarPos("Speed Thresh", "Controls"),
            'dir_threshold': cv2.getTrackbarPos("Dir Changes", "Controls"),
            'pinch_threshold': cv2.getTrackbarPos("Pinch Dist", "Controls")
        }

    def process_frame(self, frame):
        """Process a single video frame"""
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Process with MediaPipe
        results = self.hands.process(img_rgb)

        # Get parameters
        if hasattr(self, 'params'):
            params = self.params
        else:
            params = {
                'speed_threshold': 5,
                'dir_threshold': 7,
                'pinch_threshold': 30
            }

        # Process hands if detected
        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                hand_data = self.hands_data[hand_label]

                # Apply smoothing filters
                self.smooth_landmarks(landmarks, hand_label, w, h)

                # Calculate velocities and direction changes
                self.calculate_kinematics(landmarks, hand_label, hand_data, w, h)

                # Detect pinching
                self.detect_pinching(landmarks, hand_data, w, h, params['pinch_threshold'])

                # Detect finger tapping
                self.detect_tapping(hand_data)

                # Detect fidgeting
                self.detect_fidgeting(hand_data, params)

                # Draw visualization
                self.draw_visualization(frame, landmarks, hand_data, hand_label, w, h)

        # Show parameters
        cv2.putText(
            frame,
            f"Speed: {params['speed_threshold']} | Dir Changes: {params['dir_threshold']} | Pinch: {params['pinch_threshold']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return frame

    def smooth_landmarks(self, landmarks, hand_label, w, h):
        """Apply smoothing to landmark positions"""
        hand_speed = 0
        for idx, lm in enumerate(landmarks.landmark):
            x_raw, y_raw = lm.x * w, lm.y * h
            x_filtered = self.filters[hand_label][0][idx](x_raw)
            y_filtered = self.filters[hand_label][1][idx](y_raw)

            # Track filtering amount to estimate speed
            hand_speed += np.hypot(x_raw - x_filtered, y_raw - y_filtered)

            # Update landmark with filtered position
            lm.x = x_filtered / w
            lm.y = y_filtered / h

        # Store average hand speed
        self.hands_data[hand_label]['avg_speed'] = hand_speed / 21

        # Adapt filter strength based on speed
        for idx in range(21):
            # Make filter more responsive when hand is moving fast
            self.filters[hand_label][0][idx].alpha = min(0.2 + (hand_speed / 1000), 0.8)
            self.filters[hand_label][1][idx].alpha = min(0.2 + (hand_speed / 1000), 0.8)

    def calculate_kinematics(self, landmarks, hand_label, hand_data, w, h):
        """Calculate velocities and direction changes for fidget detection"""
        for tip in self.tip_ids:
            lm = landmarks.landmark[tip]
            pos = (lm.x * w, lm.y * h)

            # Calculate velocity
            if hand_data['pos_history'][tip]:
                prev_pos = hand_data['pos_history'][tip][-1]
                vel = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                speed = np.hypot(vel[0], vel[1])

                hand_data['vel_history'][tip].append((vel[0], vel[1], speed))

                # Detect direction changes
                if len(hand_data['vel_history'][tip]) > 1:
                    prev_vel = hand_data['vel_history'][tip][-2]
                    # Dot product < 0 means direction change > 90 degrees
                    if prev_vel[0] * vel[0] + prev_vel[1] * vel[1] < 0:
                        hand_data['dir_changes'][tip].append(1)
                    else:
                        hand_data['dir_changes'][tip].append(0)

            hand_data['pos_history'][tip].append(pos)

    def detect_pinching(self, landmarks, hand_data, w, h, pinch_threshold):
        """Detect pinching gestures"""
        pinch_distances = []
        for finger in self.pinch_fingers[1:]:
            thumb = landmarks.landmark[self.pinch_fingers[0]]
            finger_lm = landmarks.landmark[finger]

            dx = (thumb.x - finger_lm.x) * w
            dy = (thumb.y - finger_lm.y) * h
            distance = np.hypot(dx, dy)
            pinch_distances.append(distance)

        avg_distance = np.mean(pinch_distances)
        hand_data['pinch_dist'].append(avg_distance)

        # Detect pinch events (transitions from not pinching to pinching)
        is_pinching = avg_distance < pinch_threshold
        if is_pinching and not hand_data['last_pinch']:
            hand_data['pinch_events'].append(1)
        else:
            hand_data['pinch_events'].append(0)

        hand_data['last_pinch'] = is_pinching

    def detect_tapping(self, hand_data):
        """Detect finger tapping on surfaces"""
        # Look for rapid up-down movements in the y velocity
        # This is a simplistic implementation - can be improved
        for tip in self.tip_ids[1:]:  # Skip thumb
            if len(hand_data['vel_history'][tip]) > 2:
                vel_y_1 = hand_data['vel_history'][tip][-1][1]
                vel_y_2 = hand_data['vel_history'][tip][-2][1]

                # Check for significant direction change in vertical movement
                if vel_y_1 * vel_y_2 < 0 and abs(vel_y_1) > 5 and abs(vel_y_2) > 5:
                    hand_data['tap_events'].append(1)
                else:
                    hand_data['tap_events'].append(0)

    def detect_fidgeting(self, hand_data, params):
        """Determine if hand is fidgeting based on analyzed data"""
        # Skip if in cooldown period
        if hand_data['cooldown'] > 0:
            hand_data['cooldown'] -= 1
            return

        # Fidgeting detection based on finger movement
        movement_fidget = False
        for tip in self.tip_ids:
            if len(hand_data['dir_changes'][tip]) == self.WINDOW_LEN:
                # Count direction changes
                dir_change_count = sum(hand_data['dir_changes'][tip])

                # Calculate average speed
                speeds = [v[2] for v in hand_data['vel_history'][tip]]
                avg_speed = np.mean(speeds) if speeds else 0

                # Fidgeting = rapid direction changes + sufficient speed
                if dir_change_count > params['dir_threshold'] and avg_speed > params['speed_threshold']:
                    movement_fidget = True
                    break

        # Pinch-based fidgeting
        pinch_count = sum(hand_data['pinch_events'])
        pinch_fidget = pinch_count > self.FIDGET_THRESHOLD

        # Tap-based fidgeting
        tap_count = sum(hand_data['tap_events']) if 'tap_events' in hand_data else 0
        tap_fidget = tap_count > self.FIDGET_THRESHOLD

        # Update hand state
        if movement_fidget:
            hand_data['state'] = HandState.FIDGETING
            hand_data['cooldown'] = self.COOLDOWN_TIME
            hand_data['display_counter'] = self.DISPLAY_TIME
        elif pinch_fidget:
            hand_data['state'] = HandState.PINCHING
            hand_data['cooldown'] = self.COOLDOWN_TIME // 2
            hand_data['display_counter'] = self.DISPLAY_TIME
        elif tap_fidget:
            hand_data['state'] = HandState.TAPPING
            hand_data['cooldown'] = self.COOLDOWN_TIME // 2
            hand_data['display_counter'] = self.DISPLAY_TIME
        else:
            hand_data['state'] = HandState.NORMAL

    def draw_visualization(self, frame, landmarks, hand_data, hand_label, w, h):
        """Draw visualization elements on the frame"""
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        # Draw fidgeting state
        if hand_data['display_counter'] > 0:
            # Get position for text (near index finger)
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            text_x, text_y = int(index_tip.x * w), int(index_tip.y * h - 20)

            # Set color based on state
            if hand_data['state'] == HandState.FIDGETING:
                color = (0, 255, 0)  # Green
            elif hand_data['state'] == HandState.PINCHING:
                color = (0, 165, 255)  # Orange
            elif hand_data['state'] == HandState.TAPPING:
                color = (255, 0, 255)  # Magenta
            else:
                color = (255, 255, 255)  # White

            # Draw state text
            cv2.putText(
                frame, f"{hand_label} {hand_data['state'].name}",
                (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

            hand_data['display_counter'] -= 1

            # Optional: Draw velocities for debugging
            if hand_data['state'] != HandState.NORMAL:
                for tip in self.tip_ids:
                    if hand_data['vel_history'][tip]:
                        vel = hand_data['vel_history'][tip][-1]
                        pos = hand_data['pos_history'][tip][-1]
                        end_x = int(pos[0] + vel[0] * 5)
                        end_y = int(pos[1] + vel[1] * 5)
                        cv2.line(frame, (int(pos[0]), int(pos[1])), (end_x, end_y), color, 2)


# Main function to run the detector
def run_detector():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Initialize detector
    detector = HandFidgetDetector()
    detector.setup_ui()

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        # Update parameters
        detector.params = detector.update_params()

        # Process frame
        output_frame = detector.process_frame(frame)

        # Display result
        cv2.imshow("Hand Fidget Detector", output_frame)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()