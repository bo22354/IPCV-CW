import cv2
import numpy as np


def detectAndDisplay(frame, model):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    foundBoxes = []

    signs = model.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300)) #For circles
    # signs = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300)) #Viola Jones by itself

    for i in range(0, len(signs)): # For each detection get the start and end coordinates
        start_point = (signs[i][0], signs[i][1])
        end_point = (signs[i][0] + signs[i][2], signs[i][1] + signs[i][3])
        foundBoxes.append([start_point, end_point])

###############################################################################
#Image of Viola Jones Boxes
    # frame_cpy = np.copy(frame)
    # colour = (0, 255, 0)
    # display(foundBoxes, frame_cpy, colour)
    # cv2.imwrite( "ViolaJones.jpg", frame_cpy ) #Save Result Image
###############################################################################
    return foundBoxes


def display(foundBoxes, frame, colour, thickness = 2):
    for box in foundBoxes:
        xStart, yStart = box[0]
        xEnd, yEnd = box[1]
        cv2.rectangle(frame, (xStart, yStart), (xEnd, yEnd), colour, thickness) 


def readGroundtruth(imageName, frame):
    filename='groundtruth.txt'
    realBoxes = []

    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]

            x = int(float(content_list[1]))
            y = int(float(content_list[2]))
            width = int(float(content_list[3]))
            height = int(float(content_list[4]))

            if(img_name == imageName):
                start_point = (x, y)
                end_point = (x+width, y+height)
                realBoxes.append([start_point, end_point])
    return realBoxes


def iou(foundBoxes, realBoxes):
    xStartF, yStartF = foundBoxes[0]
    xEndF, yEndF = foundBoxes[1]
    for realBox in realBoxes:
        xStartR, yStartR = realBox[0]
        xEndR, yEndR = realBox[1]

        xLeft = max(xStartF, xStartR)
        xRight = min(xEndF, xEndR)
        yTop = max(yStartF, yStartR)
        yBottom = min(yEndF, yEndR)

        if xLeft < xRight and yTop < yBottom:
            intersect = (xRight - xLeft) * (yBottom - yTop)
            foundArea = (xEndF - xStartF) * (yEndF - yStartF)
            realArea = (xEndR - xStartR) * (yEndR - yStartR)
            union = foundArea + realArea - intersect

            iou = intersect / union
            if(iou > 0.5):
                return 1 #sufficient overlap  
    return 0 #No overlap

