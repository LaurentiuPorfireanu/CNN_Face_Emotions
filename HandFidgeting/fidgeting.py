import cv2
import numpy as np
from collections import deque
from enum import Enum
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Helpers.StatusLight import StatusLight

warnings.filterwarnings('ignore')


# Define hand states
class HandState(Enum):
    NORMAL = 0
    FIDGETING = 1
    PINCHING = 2
    TAPPING = 3


class HandFidgetWidget(ttk.Frame):
    def __init__(self, parent, width=500, height=200, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)

        self.parent = parent
        self.width = width
        self.height = height

        # Configuration parameters
        self.WINDOW_LEN = 20
        self.FIDGET_THRESHOLD = 5
        self.PINCH_DISTANCE_THRESHOLD = 30
        self.COOLDOWN_TIME = 30
        self.DISPLAY_TIME = 60

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Initialize hand landmarks
        self.tip_ids = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        self.pinch_fingers = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        # Widget state
        self.is_active = False
        self.is_loading = False
        self.hands = None
        self.is_loaded = False

        # Filters and data structures for tracking hands
        self.hands_data = {'Left': self.create_hand_data(), 'Right': self.create_hand_data()}
        self.filters = {
            'Left': [[self.ExpFilter() for _ in range(21)], [self.ExpFilter() for _ in range(21)]],
            'Right': [[self.ExpFilter() for _ in range(21)], [self.ExpFilter() for _ in range(21)]]
        }

        # For visualization
        self.frame_counter = 0
        self.result_queue = queue.Queue()
        self.last_frame_time = time.time()

        # Build UI
        self.create_ui()
        self.check_results()

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

    def create_ui(self):
        """Create the UI layout for the widget"""
        self.grid_propagate(False)
        self.config(width=self.width, height=self.height)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=3)  # Chart area
        self.grid_columnconfigure(1, weight=1)  # Status area
        self.grid_rowconfigure(0, weight=1)  # Main area
        self.grid_rowconfigure(1, weight=0)  # Control panel

        # Chart frame for visualization
        chart_frame = ttk.Frame(self)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.create_chart_frame(chart_frame)

        # Status frame for text display
        status_frame = ttk.Frame(self)
        status_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.create_status_frame(status_frame)

        # Control panel at bottom
        control_frame = ttk.Frame(self)
        control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # Modified grid configuration - all controls at left, with empty space to the right
        control_frame.grid_columnconfigure(0, weight=0)  # Load button
        control_frame.grid_columnconfigure(1, weight=0)  # Start button
        control_frame.grid_columnconfigure(2, weight=0)  # Status light
        control_frame.grid_columnconfigure(3, weight=1)  # Empty space takes remaining width

        # Move all controls to the left side
        self.load_button = ttk.Button(
            control_frame,
            text="Load Model",
            command=self.toggle_model,
            width=12
        )
        self.load_button.grid(row=0, column=0, padx=(2, 2), pady=2, sticky="w")

        self.start_button = ttk.Button(
            control_frame,
            text="Start",
            command=self.toggle_processing,
            state="disabled",
            width=12
        )
        self.start_button.grid(row=0, column=1, padx=(2, 2), pady=2, sticky="w")

        self.status_light = StatusLight(control_frame, size=15)
        self.status_light.grid(row=0, column=2, padx=(2, 5), pady=2, sticky="w")
        self.status_light.set_state("off")

    def create_chart_frame(self, parent):
        """Create visualization chart"""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        is_dark_theme = self.is_dark_theme()
        bg_color = '#2b2b2b' if is_dark_theme else 'white'
        text_color = 'white' if is_dark_theme else 'black'

        # Create figure and axes
        self.fig = Figure(figsize=(self.width * 0.7 / 100, self.height * 0.65 / 100), dpi=100)
        self.fig.patch.set_facecolor(bg_color)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.25)

        self.ax = self.fig.add_subplot(111)

        # Set up the initial plot
        self.x_data = np.arange(30)
        self.y_data_left = np.zeros(30)
        self.y_data_right = np.zeros(30)

        self.line_left, = self.ax.plot(self.x_data, self.y_data_left, 'g-', label='Left Hand')
        self.line_right, = self.ax.plot(self.x_data, self.y_data_right, 'b-', label='Right Hand')

        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 29)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True, alpha=0.3)

        # Configure chart appearance
        self.ax.set_facecolor(bg_color)
        for spine in self.ax.spines.values():
            spine.set_color(text_color)

        self.ax.tick_params(axis='both', colors=text_color, labelsize=8)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

    def create_status_frame(self, parent):
        """Create status display area"""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=0)  # For Left hand label
        parent.rowconfigure(1, weight=0)  # For Left hand state
        parent.rowconfigure(2, weight=0)  # For Right hand label
        parent.rowconfigure(3, weight=0)  # For Right hand state
        parent.rowconfigure(4, weight=1)  # For spacing

        # Left hand label
        left_label = ttk.Label(
            parent,
            text="Left Hand",
            font=("Arial", 10, "bold"),
            anchor="center"
        )
        left_label.grid(row=0, column=0, sticky="ew", pady=(5, 0))

        # Left hand state
        self.left_state_label = ttk.Label(
            parent,
            text="Normal",
            font=("Arial", 9),
            anchor="center"
        )
        self.left_state_label.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Right hand label
        right_label = ttk.Label(
            parent,
            text="Right Hand",
            font=("Arial", 10, "bold"),
            anchor="center"
        )
        right_label.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        # Right hand state
        self.right_state_label = ttk.Label(
            parent,
            text="Normal",
            font=("Arial", 9),
            anchor="center"
        )
        self.right_state_label.grid(row=3, column=0, sticky="ew", pady=(0, 10))

    def is_dark_theme(self):
        """Detect if dark theme is being used"""
        try:
            style = ttk.Style()
            bg_color = style.lookup('TFrame', 'background')

            if not bg_color:
                tmp = ttk.Frame(self)
                bg_color = tmp.winfo_rgb(tmp.cget('background'))
                tmp.destroy()
                brightness = (bg_color[0] + bg_color[1] + bg_color[2]) / (3 * 257)
                return brightness < 128
            else:
                if isinstance(bg_color, str):
                    if bg_color.startswith('#'):
                        bg_color = tuple(int(bg_color[i:i + 2], 16) for i in (1, 3, 5))
                    else:
                        tmp = ttk.Frame(self)
                        bg_color = tmp.winfo_rgb(bg_color)
                        tmp.destroy()
            brightness = sum(bg_color) / (3 * 257)
            return brightness < 128
        except:
            return False

    def toggle_model(self):
        """Toggle model loading/unloading"""
        if self.is_loading:
            return

        # Prevent unloading while active
        if self.is_loaded and self.is_active:
            return

        if not self.is_loaded:
            self.is_loading = True
            self.load_button.config(state="disabled")
            self.status_light.set_state("loading")

            def load_model_thread():
                try:
                    # Create Hands model - this will be loaded when needed
                    self.hands = self.mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7
                    )
                    self.is_loaded = True
                    self.result_queue.put(("model_loaded", None))
                except Exception as e:
                    print(f"Error loading model: {e}")
                    self.result_queue.put(("model_load_failed", None))
                self.is_loading = False

            threading.Thread(target=load_model_thread).start()
        else:
            if self.hands:
                self.hands.close()
                self.hands = None
            self.is_loaded = False
            self.load_button.config(text="Load Model")
            self.start_button.config(state="disabled")
            self.status_light.set_state("off")
            self.clear_display()

    def toggle_processing(self):
        """Toggle hand processing on/off"""
        if not self.is_loaded or self.is_loading:
            return

        if not self.is_active:
            self.is_active = True
            self.start_button.config(text="Stop")
            self.status_light.set_state("active")
        else:
            self.is_active = False
            self.start_button.config(text="Start")
            self.status_light.set_state("ready")
            self.clear_display()

    def process_frame(self, frame):
        """Process a video frame"""
        if not self.is_active or not self.is_loaded:
            return frame

        # Throttle processing to avoid overloading CPU
        current_time = time.time()
        if current_time - self.last_frame_time > 0.1:  # Process at most every 100ms
            self.last_frame_time = current_time

            # Process frame in a separate thread to avoid blocking UI
            threading.Thread(
                target=self.process_frame_thread,
                args=(frame.copy(),)
            ).start()

        return frame

    def process_frame_thread(self, frame):
        """Process frame for hand tracking and fidget detection"""
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Process with MediaPipe
            results = self.hands.process(frame_rgb)

            # Default values
            left_state = HandState.NORMAL
            right_state = HandState.NORMAL
            left_activity = 0
            right_activity = 0

            # Process hands if detected
            if results.multi_hand_landmarks:
                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    hand_data = self.hands_data[hand_label]

                    # Apply smoothing filters
                    hand_speed = self.smooth_landmarks(landmarks, hand_label, w, h)

                    # Calculate kinematics for fidget detection
                    self.calculate_kinematics(landmarks, hand_label, hand_data, w, h)

                    # Detect pinching
                    self.detect_pinching(landmarks, hand_data, w, h, self.PINCH_DISTANCE_THRESHOLD)

                    # Detect finger tapping
                    self.detect_tapping(hand_data)

                    # Detect fidgeting
                    self.detect_fidgeting(hand_data, {
                        'speed_threshold': 5,
                        'dir_threshold': 7,
                        'pinch_threshold': 30
                    })

                    # Store state for visualization
                    if hand_label == 'Left':
                        left_state = hand_data['state']
                        left_activity = hand_speed
                    else:
                        right_state = hand_data['state']
                        right_activity = hand_speed

            # Update queue with results
            self.result_queue.put(("hand_state", (left_state, right_state, left_activity, right_activity)))

        except Exception as e:
            print(f"Error processing hand frame: {e}")

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

        return hand_speed

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

        # HandFidgeting detection based on finger movement
        movement_fidget = False
        for tip in self.tip_ids:
            if len(hand_data['dir_changes'][tip]) == self.WINDOW_LEN:
                # Count direction changes
                dir_change_count = sum(hand_data['dir_changes'][tip])

                # Calculate average speed
                speeds = [v[2] for v in hand_data['vel_history'][tip]]
                avg_speed = np.mean(speeds) if speeds else 0

                # HandFidgeting = rapid direction changes + sufficient speed
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

    def check_results(self):
        """Check and process results from the result queue"""
        try:
            while True:
                msg_type, msg_data = self.result_queue.get_nowait()

                if msg_type == "model_loaded":
                    self.load_button.config(text="Unload Model", state="normal")
                    self.start_button.config(state="normal")
                    self.status_light.set_state("ready")

                elif msg_type == "model_load_failed":
                    self.load_button.config(text="Load Model", state="normal")
                    self.status_light.set_state("off")

                elif msg_type == "hand_state":
                    left_state, right_state, left_activity, right_activity = msg_data
                    self.update_display(left_state, right_state, left_activity, right_activity)

        except queue.Empty:
            pass

        self.after(50, self.check_results)

    def update_display(self, left_state, right_state, left_activity, right_activity):
        """Update the visualization with new data"""
        # Update state labels
        self.left_state_label.config(text=left_state.name)
        self.right_state_label.config(text=right_state.name)

        # Use different colors for different states
        left_color = self.get_state_color(left_state)
        right_color = self.get_state_color(right_state)

        # Update activity chart
        self.y_data_left = np.roll(self.y_data_left, -1)
        self.y_data_left[-1] = min(100, left_activity / 10)

        self.y_data_right = np.roll(self.y_data_right, -1)
        self.y_data_right[-1] = min(100, right_activity / 10)

        self.line_left.set_ydata(self.y_data_left)
        self.line_left.set_color(left_color)
        self.line_right.set_ydata(self.y_data_right)
        self.line_right.set_color(right_color)

        self.canvas.draw_idle()

    def get_state_color(self, state):
        """Get color based on hand state"""
        if state == HandState.NORMAL:
            return 'green'
        elif state == HandState.FIDGETING:
            return 'orange'
        elif state == HandState.PINCHING:
            return 'red'
        elif state == HandState.TAPPING:
            return 'purple'
        else:
            return 'gray'

    def clear_display(self):
        """Reset display to default state"""
        # Reset state labels
        self.left_state_label.config(text="Normal")
        self.right_state_label.config(text="Normal")

        # Reset activity chart
        self.y_data_left = np.zeros(30)
        self.y_data_right = np.zeros(30)

        self.line_left.set_ydata(self.y_data_left)
        self.line_left.set_color('green')
        self.line_right.set_ydata(self.y_data_right)
        self.line_right.set_color('blue')

        self.canvas.draw_idle()


# For testing the widget independently
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Hand Fidget Detector")
    root.geometry("500x200")

    fidget_widget = HandFidgetWidget(root, width=500, height=200)
    fidget_widget.pack(fill=tk.BOTH, expand=True)

    root.mainloop()