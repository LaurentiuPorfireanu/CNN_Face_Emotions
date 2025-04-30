import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time

IMAGE_SIZE = 96
COARSE_EMOTIONS = ['happy', 'fear', 'angry', 'other']
FINE_EMOTIONS   = ['neutral', 'disgust', 'sad', 'surprise']

class EmotionCNN(nn.Module):
	"""
	A Convolutional Neural Network (CNN) for emotion classification from grayscale facial images.

	Architecture Overview:
	- 3 convolutional blocks (Conv2D + ReLU + MaxPooling + Dropout)
	- Fully connected layers for feature integration and final classification.

	Args:
		num_classes (int): Number of emotion categories to classify. Default is 7.
	"""

	def __init__(self, num_classes: int = 7) -> None:
		super(EmotionCNN, self).__init__()

		# First convolutional block
		self.conv1: nn.Conv2d = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
		self.pool1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout1: nn.Dropout = nn.Dropout(0.2)

		# Second convolutional block
		self.conv2: nn.Conv2d = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.pool2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout2: nn.Dropout = nn.Dropout(0.2)

		# Third convolutional block
		self.conv3: nn.Conv2d = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.pool3: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout3: nn.Dropout = nn.Dropout(0.2)

		# Compute the number of flattened features dynamically
		self.flat_features: int = 512 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)
		# Explanation: After 3 times pooling with stride 2, size = original_size / (2^3) = original_size / 8

		# Fully connected layers
		self.fc1: nn.Linear = nn.Linear(self.flat_features, 512)
		self.dropout4: nn.Dropout = nn.Dropout(0.2)
		self.fc2: nn.Linear = nn.Linear(512, 256)
		self.dropout5: nn.Dropout = nn.Dropout(0.2)
		self.fc3: nn.Linear = nn.Linear(256, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Defines the forward pass of the network.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, 1, 96, 96).

		Returns:
			torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
		"""
		# Apply first convolutional block
		x = F.relu(self.conv1(x))
		x = self.pool1(x)
		x = self.dropout1(x)

		# Apply second convolutional block
		x = F.relu(self.conv2(x))
		x = self.pool2(x)
		x = self.dropout2(x)

		# Apply third convolutional block
		x = F.relu(self.conv3(x))
		x = self.pool3(x)
		x = self.dropout3(x)

		# Flatten the tensor for fully connected layers
		x = x.view(-1, self.flat_features)

		# Pass through fully connected layers with dropout
		x = F.relu(self.fc1(x))
		x = self.dropout4(x)
		x = F.relu(self.fc2(x))
		x = self.dropout5(x)
		x = self.fc3(x)  # Final layer outputs raw logits

		return x
     

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform: transforms.Compose = transforms.Compose([
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images to (96, 96)
	transforms.ToTensor(),                         # Convert images to PyTorch tensors
	transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize tensors to range [-1, 1]
])


def realtime_emotion_detection(threshold: float = 0.6) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    model_coarse = EmotionCNN(num_classes=len(COARSE_EMOTIONS)).to(device)
    model_fine   = EmotionCNN(num_classes=len(FINE_EMOTIONS)).to(device)
    model_coarse.load_state_dict(torch.load('coarse_model.pth', map_location=device))
    model_fine  .load_state_dict(torch.load('fine_model.pth',   map_location=device))
    model_coarse.eval()
    model_fine.eval()
    print("Both models loaded!")

    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)
    prev_time = 0
    frame_count = 0
    current_emotion = "unknown"
    current_confidence = 0.0
    face_locations = []

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        
        curr_t = time.time()
        fps = 1/(curr_t - prev_time) if prev_time else 30
        prev_time = curr_t

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count = (frame_count + 1) % 15

        if frame_count == 0:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            face_locations = faces[:1]  

            if len(face_locations):
                x, y, w, h = face_locations[0]
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                face = Image.fromarray(face)
                tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                   
                    out_c   = model_coarse(tensor)
                    probs_c = F.softmax(out_c, dim=1)[0]
                    p_other = probs_c[COARSE_EMOTIONS.index('other')]

                    if p_other > threshold:
                        
                        out_f   = model_fine(tensor)
                        probs_f = F.softmax(out_f, dim=1)[0]
                        idx_f   = torch.argmax(out_f, dim=1).item()
                        current_emotion    = FINE_EMOTIONS[idx_f]
                        current_confidence = probs_f[idx_f].item()
                    else:
                     
                        top3 = probs_c[:3]
                        idx_c = torch.argmax(top3).item()
                        current_emotion    = COARSE_EMOTIONS[idx_c]
                        current_confidence = top3[idx_c].item()

        
        cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/15", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        for (x, y, w, h) in face_locations:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            text = f"{current_emotion}: {current_confidence:.2f}"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow('Real-time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_emotion_detection(threshold=0.3)