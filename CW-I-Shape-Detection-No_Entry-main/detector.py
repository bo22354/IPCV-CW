import numpy as np
import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
args = parser.parse_args()

# /** Global variables */
cascade_name = "NoEntrycascade/cascade.xml"

def detectAndDisplay(frame):
	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, flags=0, minSize=(20,20), maxSize=(300,300))
    # 4. Draw box around faces found
    foundBoxes = []
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
        foundBoxes.append([start_point, end_point])

    return foundBoxes
        




# ************ NEED MODIFICATION ************
def readGroundtruth(imageName, frame):
    filename='groundtruth.txt'
    # read bounding boxes as ground truth
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
            # print(img_name +' '+str(x)+' '+str(y)+' '+str(width)+' '+str(height))
            if(img_name == imageName):
                start_point = (x, y)
                end_point = (x+width, y+height)
                colour = (0,0,255)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
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

        if xLeft < xRight or yTop < yBottom:
            intersect = (xRight - xLeft) * (yBottom - yTop)
            foundArea = (xEndF - xStartF) * (yEndF - yEndR)
            realArea = (xEndR - xStartR) * (yEndR - yStartR)
            union = foundArea + realArea - intersect

            iou = intersect / union
            if(iou > 0.5):
                return 1 #sufficient overlap
    
    return 0 #No overlap
        
        
                













# ==== MAIN ==============================================

imageName = args.name

# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

fileNames = imageName.split("/")
file = fileNames[len(fileNames)- 1]
fileName = file.split(".")

# 1. Read Input Image
frame = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)


# 3. Detect Faces and Display Result
foundBoxes = detectAndDisplay( frame )
realBoxes = readGroundtruth( fileName[0], frame )
truePos = 0
for foundBox in foundBoxes:
    truePos = iou(foundBox, realBoxes) + truePos
    # print("FoundBox: " , ' ' , foundBox , ' ' , "True Positives: " , ' ' , truePos , '\n\n')


falsePos = len(foundBoxes) - truePos 
falseNeg = len(realBoxes) - truePos

if truePos + falsePos > 0:
    precision = truePos / (truePos + falsePos) #Normal situation with some detection
else:
    precision = 0 #No detections

if truePos + falseNeg > 0:
    recall = truePos / (truePos + falseNeg)
else:
    recall = 0 #Nothing to detect

if precision + recall > 0:
    f1Score = 2 * ((precision*recall) / (precision+recall))
else:
    f1Score = 0 #Nothing to be detected and it didn't detect anything

if len(realBoxes) > 0:
    truePosRate = truePos / len(realBoxes)
else:
    truePosRate = 0 #Nothing to detect


# print("Precision ", precision)
# print("Recall: ", recall)
print("True Positive Rate: ", truePosRate)
print("F1-Score: ", f1Score)


# 4. Save Result Image
cv2.imwrite( "detected.jpg", frame )


