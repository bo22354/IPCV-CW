import cv2
import numpy as np

def gradient(gradX, gradY):
    gradient = np.zeros([gradX.shape[0], gradX.shape[1]], dtype=np.uint8)
    for i in range(gradX.shape[0]):
        for j in range(gradX.shape[1]):
            gradient[i, j] = np.sqrt(gradX[i, j]**2 + gradY[i, j]**2)
    return gradient


def applyFilters(frame, filters):
    size = []
    output = []
    filterPadding = []
    paddedImage = []
    n = 0
    for filter in filters:
        size.append(len(filter[0]))
        output.append(np.zeros([frame.shape[0], frame.shape[1]], dtype=np.float32))
        filterPadding.append(round(( size[n] - 1 ) / 2))
        paddedImage.append(cv2.copyMakeBorder(frame, filterPadding[n], filterPadding[n], filterPadding[n], filterPadding[n], cv2.BORDER_REPLICATE))
        n += 1
    patch = []
    for y in range(0, frame.shape[0]):
        for x in range(0, frame.shape[1]):
            for a in range(len(filters)):
                patch = paddedImage[a][y:y+size[a], x:x+size[a]]
                filterValue = np.multiply(patch, filters[a]).sum()
                output[a][y, x] = filterValue
    return output


# def applyFilter(frame, filter):
#     size  = len(filter[0])
#     output = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.float32)
#     filterPaddingX = round(( size - 1 ) / 2)
#     filterPaddingY = round(( size - 1 ) / 2)

#     paddedImage =cv2.copyMakeBorder(frame, filterPaddingX, filterPaddingX, filterPaddingY, filterPaddingX, cv2.BORDER_REPLICATE)

#     for y in range(0, frame.shape[0]):
#         for x in range(0, frame.shape[1]):
#             patch = paddedImage[y:y+size, x:x+size]
#             filterValue = np.multiply(patch, filter).sum()
#             output[y, x] = filterValue
#     return output


def threshold(magnitude):
    t = np.percentile(magnitude, 90)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if(magnitude[i, j] > t):
                magnitude[i, j] = 255
            else:
                magnitude[i, j] = 0
    return magnitude


def hough(thresImage, angle, t, rMax, rMin, image):
    rows, cols = thresImage.shape
    accumulator = np.zeros((rows, cols, rMax - rMin), dtype=np.int32) #3d array to collect "votes" for centres of circles at different radius

    for i in range(rows):
        for j in range(cols):
            if(thresImage[i, j] == 255): #if a valid gradient to possibly be part of a circle
                for r in range(rMin, rMax): # Calcuate positions in hough space and add a vote to the point for each radius
                    y = i + r* np.sin(angle[i,j])
                    x = j + r* np.cos(angle[i,j])
                    yNeg = i - r* np.sin(angle[i,j])
                    xNeg = j + r* np.cos(angle[i,j])
                    if 0 <= round(y) < rows and 0 <= round(x) < cols: #ensure it's a real point in the image
                        accumulator[int(round(y)), int(round(x)), r - rMin] += 1
                    if 0 <= round(yNeg) < rows and 0 <= round(xNeg) < cols: #ensure it's a real point in the image
                        accumulator[int(round(yNeg)), int(round(xNeg)), r - rMin] += 1                 
    print("Got Votes")
    cv2.imwrite("circles.jpg", image)
    centres = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    circles = []
    for a in range(rows):
        for b in range(cols):
            for r in range(rMin, rMax):
                rIndex = r - rMin
                if accumulator[a, b, rIndex] >= t:
                    # circles.append([a, b, r+rMin])
                    cv2.circle(image, (b, a), r, (0, 0, 255, 4))
                    centres[a, b] = 255
    cv2.imwrite("circles.jpg", image)
    cv2.imwrite("centres.jpg", centres)
    # return circles

    
def main(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    gaussianX = cv2.getGaussianKernel(5,1)
    gaussianY = cv2.getGaussianKernel(5,1)
    gaussian = gaussianX * gaussianY.T
    
    sobelX = [[-1,0,1], 
           [-2,0,2],
           [-1,0,1]]
    
    sobelY = [[-1,-2,-1], 
           [0,0,0],
           [1,2,1]]
    smoothed = applyFilters(frame_gray, [gaussian])[0]
    print("Got Guassian")
    gradX, gradY = applyFilters(smoothed, [sobelX, sobelY])
    print("Got Grads")
    magnitude = cv2.magnitude(gradX, gradY)
    print("Got Mag")
    angle = cv2.phase(gradX, gradY, angleInDegrees=False)
    print("Got Angle")
    cv2.imwrite("angle.jpg", angle)
    # angle_normalized = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #For showing angle image in degrees
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #normalize(src, destination, min, max, normalization type, data type)
    thresImage = threshold(magnitude)
    print("Got Threshold Image")
    cv2.imwrite("thresImage.jpg", thresImage)

    t = 10
    rMin = 40
    rMax = 100
    circles = hough(thresImage, angle, t, rMax, rMin, frame)
    print(circles)
    return 1


def temp():
    angle = cv2.imread("angle.jpg", 1)
    thresImage = cv2.imread("thresImage.jpg", 1)
    thresImage = cv2.cvtColor(thresImage, cv2.COLOR_BGR2GRAY)
    thresImage = cv2.equalizeHist(thresImage)
    angle = cv2.cvtColor(angle, cv2.COLOR_BGR2GRAY)
    angle = cv2.equalizeHist(angle)
    print(thresImage.shape)
    t = 2
    rMin = 50
    rMax = 100
    circles = hough(thresImage, angle, t, rMax, rMin)
    print(circles)


    




