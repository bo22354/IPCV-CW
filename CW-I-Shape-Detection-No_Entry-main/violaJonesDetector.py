import cv2



def detectAndDisplay(frame, model):
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