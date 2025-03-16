import cv2
from scipy.spatial import distance as dist
import smtplib

# Define functions for eye aspect ratio (EAR) and mouth aspect ratio (MAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set thresholds and frame counters
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to send email alerts
def send_alert(condition):
    sender_email = "youremail@example.com"
    receiver_email = "receiver@example.com"
    password = "yourpassword"
    
    subject = f"Alert: {condition} detected!"
    body = f"The driver is {condition.lower()}."
    
    message = f'Subject: {subject}\n\n{body}'
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Region of interest for the face
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_gray)
        eye_list = []

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            eye = face_gray[ey:ey + eh, ex:ex + ew]
            eye_list.append([ex, ey, ex + ew, ey + eh])
        
        if len(eye_list) >= 2:  # Need at least two eyes for EAR calculation
            left_eye = eye_list[0]
            right_eye = eye_list[1]
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Detect drowsiness
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    send_alert("Drowsy")
            else:
                COUNTER = 0
                cv2.putText(frame, "Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Driver Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
