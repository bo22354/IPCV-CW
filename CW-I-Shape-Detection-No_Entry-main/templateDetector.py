import cv2
import numpy as np


def rotateTemplate(template, angle):
    h, w = template.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(template, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_image



def templateMatching(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    scales = np.linspace(0.04, 0.5, 100)
    angles = range(-15, 16, 5)
    templateH, templateW = template_gray.shape
    boxes = []

    for scale in scales: #Does template matching on each of the scales
        scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        for angle in angles:
            rotated_template = rotateTemplate(scaled_template, angle)
            templateH, templateW = rotated_template.shape


            result = cv2.matchTemplate(image_gray, rotated_template, method=cv2.TM_CCOEFF_NORMED)

            valid = np.where(result >= 0.5) #get the areas only with a result above the threshold
            for area in zip(*valid): #gets the x and y for each valid area
                y, x = area
                # boxes.append((x, y, x + int(templateW * scale), y + int(templateH * scale), result[y, x]))
                boxes.append((x, y, x + templateW, y + templateH, result[y, x]))

    return boxes


def nonMaxSuppression(boxes):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    order = scores.argsort()[::-1] # Sort to have the highest matching box first
    areas = (x2 - x1) * (y2 - y1) # get the areas of all the boxes
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU with the remaining boxes
        xRight = np.maximum(x1[i], x1[order[1:]])
        yBottom = np.maximum(y1[i], y1[order[1:]])
        xLeft = np.minimum(x2[i], x2[order[1:]])
        yTop = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xLeft - xRight)
        h = np.maximum(0, yTop - yBottom)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union

        order = order[np.where(iou <= 0.3)[0] + 1] #find all the boxes with an iou values <= 0.5 and if so then it's not detecting the same feature
    return boxes[keep].astype(int)


def main(image):
    template_image = cv2.imread("no_entry.jpg")
    boxes = templateMatching(image, template_image) #gets all the boxes that matches
    filtered_boxes = nonMaxSuppression(boxes) #filters out boxes using
    boxes = [[box[0], box[1], box[2], box[3], box[4]] for box in filtered_boxes]
    print("Done Matching")
    return boxes


    # Draw the bounding boxes on the image
    for box in filtered_boxes:
        x1, y1, x2, y2, __name__ = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save or display the resulting image
    # cv2.imwrite("result.jpg", image)
    cv2.imshow("Detected Signs", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# img = cv2.imread("No_entry/NoEntry5.bmp")
# main(img)