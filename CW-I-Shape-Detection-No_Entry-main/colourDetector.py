import cv2
import numpy as np




def detectColours(image, boxes):
    

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert image to HSV
    
    # Define HSV ranges for red
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    # Define HSV ranges for white
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 50, 255])
    
    valid_boxes = []
    
    for box in boxes:
        startX, startY = box[0]
        endX, endY = box[1]
        area = hsv[startY:endY, startX:endX] # gets the area in the HSV image
        
        # Create masks for red and white
        red_mask1 = cv2.inRange(area, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(area, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # white_mask = cv2.inRange(area, white_lower, white_upper)
        
        # Calculate percentages
        total_pixels = area.shape[0] * area.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        # white_pixels = cv2.countNonZero(white_mask)
        
        red_percentage = (red_pixels / total_pixels) * 100
        # white_percentage = (white_pixels / total_pixels) * 100

        print("Red: ", red_percentage)
        # print("White :", white_percentage)
        
        # Check thresholds
        if red_percentage >= 15:
            valid_boxes.append(box)
    
    return valid_boxes