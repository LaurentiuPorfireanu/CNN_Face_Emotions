import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import time
IMAGE_SIZE = 96



# Define the model architecture (identical to the one in your trainmodel.ipynb)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # Calculate the size after convolutions
        self.flat_features = 512 * 12 * 12

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(-1, self.flat_features)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)

        return x


# Labels - must match the ones used in training
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Same image preprocessing as in training
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Main function for real-time emotion detection
def realtime_emotion_detection():
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = EmotionCNN().to(device)
    try:
        model.load_state_dict(torch.load('PyTorch/emotion_model.pth', map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()  # Set model to evaluation mode

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face detector cascade")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")

    # For frame rate calculation
    prev_time = 0

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 30
        prev_time = current_time

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Draw FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            # Extract face region
            face_region = gray[y:y + h, x:x + w]

            # Resize to model's input size
            face_resized = cv2.resize(face_region, (IMAGE_SIZE, IMAGE_SIZE))

            # Convert to PIL Image
            face_pil = Image.fromarray(face_resized)

            # Apply transformations
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

            # Get emotion label and confidence
            emotion = EMOTIONS[prediction.item()]
            conf_value = confidence.item()

            # Draw rectangle and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotion_text = f"{emotion}: {conf_value:.2f}"

            # Determine text color based on emotion
            color = (255, 255, 255)  # Default: white
            if emotion == "happy":
                color = (0, 255, 255)  # Yellow
            elif emotion == "angry":
                color = (0, 0, 255)  # Red
            elif emotion == "sad":
                color = (255, 0, 0)  # Blue
            elif emotion == "surprise":
                color = (0, 255, 0)  # Green

            # Display emotion and confidence
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_emotion_detection()