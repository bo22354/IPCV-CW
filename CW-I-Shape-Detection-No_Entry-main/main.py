import numpy as np
import cv2
import os
import sys
import argparse
import violaJonesDetector
import shapeDetector
import templateDetector
import colourDetector


parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='No_entry/NoEntry0.bmp')
parser.add_argument('-type', '-t', type=str, default='all')

args = parser.parse_args()
cascade_name = "NoEntrycascade/cascade.xml"


def analysis():
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

    #Analysis Outputs
    # print("True Positive: ",truePos)
    # print("False Positives: ", falsePos)
    # print("False Negatives: ", falseNeg)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1: ", f1Score)
    # print("TPR:  ", truePosRate)


    
    return [falsePos, falseNeg, precision, recall, f1Score, truePosRate]

    



imageName = args.name

if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)): # ignore if no such file is present.
    print('No such file')
    sys.exit(1)

fileNames = imageName.split("/")
file = fileNames[len(fileNames)- 1]
fileName = file.split(".") #get actual file name (no path or anytrhing other than the name)

frame = cv2.imread(imageName, 1) #Read Input Image

if not (type(frame) is np.ndarray): # ignore if image is not array.
    print('Not image data')
    sys.exit(1)


model = cv2.CascadeClassifier() #Load the Strong Classifier in a structure called `Cascade'
if not model.load(cascade_name): 
    print('--(!)Error loading cascade model')
    exit(0)


templateBoxes = []
templateBoxes = templateDetector.main(frame)
# print("Found Via Template Matching")

violaBoxes = violaJonesDetector.detectAndDisplay( frame , model ) #Detect NoEntry signs and display
# print("Found Via ViolaJones")


violaBoxes = [[box[0][0], box[0][1], box[1][0], box[1][1], 0.6] for box in violaBoxes]
foundBoxes = violaBoxes + templateBoxes
foundBoxes = templateDetector.nonMaxSuppression(foundBoxes)

foundBoxes = [[[box[0], box[1]], [box[2], box[3]]] for box in foundBoxes]
# print("Done NMS of boxes")

###############################################################################
#Image of all Boxes Created
# allBoxes = np.copy(frame)
# violaJonesDetector.display(foundBoxes, allBoxes, (0, 255, 0))
# cv2.imwrite( "allBoxes.jpg", allBoxes)#Save Result Image
###############################################################################



if args.type == 'all':
    circleBoxes = shapeDetector.main(frame)
    # print("Found Circles")
    newFoundBoxes = []
    for foundBox in foundBoxes:
        if violaJonesDetector.iou(foundBox, circleBoxes) == 1:
            newFoundBoxes.append(foundBox)
    foundBoxes = newFoundBoxes
    # print("Finished Detecting")


#Check all boxes for colour matching
foundBoxes = colourDetector.detectColours(frame, foundBoxes)


realBoxes = violaJonesDetector.readGroundtruth( fileName[0], frame ) #Displasy the groundtruth values on image
violaJonesDetector.display(foundBoxes, frame, (0, 255, 0), 4)
violaJonesDetector.display(realBoxes, frame, (0, 0, 255))


truePos = 0
for foundBox in foundBoxes: #determine if each found box is a valid detection
    truePos = violaJonesDetector.iou(foundBox, realBoxes) + truePos


results = analysis()

# print("True Positive Rate: ", results[5])
# print("F1-Score: ", results[4])

cv2.imwrite( "detected.jpg", frame )#Save Result Image
