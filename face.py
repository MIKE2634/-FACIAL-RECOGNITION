import cv2

# Correct absolute path to the Haar Cascade XML file
cascade_path = r'C:\Users\Admin\Desktop\familirization\haarcascade_frontalface_default.xml'

# Load the cascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade classifier is loaded successfully
if face_cascade.empty():
    print("Error loading cascade classifier")
else:
    print("Cascade classifier loaded successfully")

# Initialize the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
