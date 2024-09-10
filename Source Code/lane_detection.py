import cv2
import numpy as np

# Function to process the frame and detect lane lines
def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection (Canny)
    edges = cv2.Canny(blur, 50, 150)
    
    # Region of interest (Mask)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(100, height), (width-100, height), (width//2, height//2)]])
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
    
    # Drawing lines on the frame
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    # Combine the result with the original frame
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return final_image

# Video capture from file or webcam
cap = cv2.VideoCapture('solidWhiteRight.mp4')#Replace solidWhiteRight.mp4 with a path to your video file or use cv2.VideoCapture(0) for webcam input.

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_frame(frame)
    
    # Display the result
    cv2.imshow('Lane Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
