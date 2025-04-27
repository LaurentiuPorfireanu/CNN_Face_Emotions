import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time

IMAGE_SIZE = 96


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


def realtime_emotion_detection() -> None:
	"""
	Runs real-time emotion detection using a webcam feed.

	Steps:
	- Load the pre-trained EmotionCNN model.
	- Detect faces in real-time using OpenCV Haar Cascades.
	- Predict emotions for detected faces every 15 frames.
	- Display the predicted emotion, confidence score, and FPS on the video feed.

	Returns:
		None
	"""
	
	device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")

	
	model: EmotionCNN = EmotionCNN().to(device)
	try:
		model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
		print("Model loaded successfully!")
	except Exception as e:
		print(f"Error loading model: {e}")
		return

	model.eval()  

	# Load Haar Cascade face detector
	face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	if face_cascade.empty():
		print("Error: Could not load face detector cascade")
		return

	# Initialize webcam
	cap: cv2.VideoCapture = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam")
		return

	print("Press 'q' to quit")

	# Initialize variables for FPS calculation
	prev_time: float = 0.0

	# Variables for frame counting and emotion detection
	frame_count: int = 0
	current_emotion: str = "unknown"
	current_confidence: float = 0.0
	face_locations: list = []

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Error: Failed to capture image")
			break

		# Calculate FPS
		current_time: float = time.time()
		fps: float = 1 / (current_time - prev_time) if prev_time > 0 else 30
		prev_time = current_time

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Update frame counter
		frame_count = (frame_count + 1) % 15

		# Perform face detection and emotion prediction every 15 frames
		if frame_count == 0:
			faces = face_cascade.detectMultiScale(
				gray,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(30, 30)
			)
			face_locations = faces

			# Process only the first detected face
			if len(faces) > 0:
				(x, y, w, h) = faces[0]

				# Extract the face region
				face_region = gray[y:y + h, x:x + w]

				# Resize face to model input size
				face_resized = cv2.resize(face_region, (IMAGE_SIZE, IMAGE_SIZE))

				# Convert to PIL image
				face_pil = Image.fromarray(face_resized)

				face_tensor = transform(face_pil).unsqueeze(0).to(device)

				with torch.no_grad():
					outputs = model(face_tensor)
					probabilities = F.softmax(outputs, dim=1)
					confidence, prediction = torch.max(probabilities, 1)

				# Update current emotion and confidence
				current_emotion = EMOTIONS[prediction.item()]
				current_confidence = confidence.item()


		cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)


		cv2.putText(frame, f"Frame: {frame_count}/15", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)


		for (x, y, w, h) in face_locations:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			emotion_text: str = f"{current_emotion}: {current_confidence:.2f}"

			# Set text color based on detected emotion
			color: tuple = (255, 255, 255) 
			if current_emotion == "happy":
				color = (0, 255, 255)
			elif current_emotion == "angry":
				color = (0, 0, 255) 
			elif current_emotion == "sad":
				color = (255, 0, 0) 
			elif current_emotion == "surprise":
				color = (0, 255, 0) 

			# Draw emotion text above face rectangle
			cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
						0.9, color, 2)

		# Display the resulting frame
		cv2.imshow('Real-time Emotion Detection', frame)

	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release resources
	cap.release()
	cv2.destroyAllWindows()
	


if __name__ == "__main__":
    realtime_emotion_detection()