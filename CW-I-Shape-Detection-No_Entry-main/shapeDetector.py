import cv2
import numpy as np
import matplotlib.pyplot as plt


def applyFilters(frame, filters):
    size = []
    output = []
    filterPadding = []
    paddedImage = []
    patch = []
    n = 0

    for filter in filters: 
        size.append(len(filter[0])) #radius of the filter
        output.append(np.zeros([frame.shape[0], frame.shape[1]], dtype=np.float32)) #create new array for img after filter is applied
        filterPadding.append(round(( size[n] - 1 ) / 2)) #Calculate the padding needed around the image to be able to apply filter to the whole image
        paddedImage.append(cv2.copyMakeBorder(frame, filterPadding[n], filterPadding[n], filterPadding[n], filterPadding[n], cv2.BORDER_REPLICATE))
        n += 1

    #Go through each pixel in the image and apply each filter and save to the corresponding array
    for y in range(0, frame.shape[0]):
        for x in range(0, frame.shape[1]):
            for a in range(len(filters)):
                patch = paddedImage[a][y:y+size[a], x:x+size[a]]
                filterValue = np.multiply(patch, filters[a]).sum()
                output[a][y, x] = filterValue
    return output


def threshold(magnitude): 
    t = 100

    #For each pixel check if over threshold if so change to max otherwise set as 0
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if(magnitude[i, j] > t):
                magnitude[i, j] = 255
            else:
                magnitude[i, j] = 0
    return magnitude


def hough(thresImage, angle, t, rMax, rMin, image):
    rows, cols = thresImage.shape
    accumulator = np.zeros((rows, cols, rMax - rMin), dtype=np.int32)  # 3D accumulator

    for i in range(rows):
        for j in range(cols):
            theta = angle[i, j]
            if thresImage[i, j] > 0:  # If there's a valid gradient to possibly be part of a circle
                for r in range(rMin, rMax): #for all radius in the range find the x and y values (positive and negative)
                    y = int(i + r * np.sin(theta))
                    x = int(j + r * np.cos(theta))
                    yNeg = int(i - r * np.sin(theta))
                    xNeg = int(j - r * np.cos(theta)) 

                    # Ensure the coordinates are within bounds and add votes
                    if 0 <= y < rows and 0 <= x < cols:
                        accumulator[y, x, r - rMin] += 1
                    if 0 <= yNeg < rows and 0 <= xNeg < cols:
                        accumulator[yNeg, xNeg, r - rMin] += 1

    img_cpy = np.copy(image)
    circlesBoxes = []
    # houghSpace = np.zeros((rows, cols), dtype=np.int32) #needed for plotting hough space

    #Find circles above threshold
    for a in range(rows):
        for b in range(cols):
            count = 0
            for r in range(rMin, rMax):
                rIndex = r - rMin
                # count += accumulator[a, b, rIndex] #needed for plotting hough space
                if accumulator[a, b, rIndex] >= t:
                    circlesBoxes.append([[b-r, a-r],[b+r, a+r]])  # Add the coordinates for the box bounding the circle
                    cv2.circle(img_cpy, (b, a), r, (255, 0, 0), 4)  # Draw circle on image
            # houghSpace[a, b] = count #needed for plotting hough space
    

#################################################################
#Plotting Hough Space
    # plt.imshow(houghSpace, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Votes')
    # plt.xlabel('Theta')
    # plt.ylabel('Rho')
    # plt.title('2D Hough Space (Radii summed)')
    # plt.show()
    # cv2.imwrite("houghSpace.jpg", houghSpace)
#################################################################


    # cv2.imwrite("circles.jpg", img_cpy)
    return circlesBoxes


def main(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    ############################################################################
    # Using cv2 
    ###########################################################################
    # gaussian_img = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # # Step 2: Apply Sobel operator to get gradients in x and y directions
    # sobel_x = cv2.Sobel(gaussian_img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    # sobel_y = cv2.Sobel(gaussian_img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

    # # Step 3: Compute the magnitude and angle of the gradient
    # magnitude = cv2.magnitude(sobel_x, sobel_y)  # Magnitude = sqrt(sobel_x^2 + sobel_y^2)
    # angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=False)

    # thresImage = threshold(magnitude)
    # _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    # cv2.imwrite("thresImage.jpg", edges)
    #########################################################################


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
    gradX, gradY = applyFilters(smoothed, [sobelX, sobelY])
    magnitude = cv2.magnitude(gradX, gradY)
    angle = cv2.phase(gradX, gradY, angleInDegrees=False)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #normalize(src, destination, min, max, normalization type, data type)
    thresImage = threshold(magnitude)

    t = 13
    rMin = 10
    rMax = 110
    circles = hough(thresImage, angle, t, rMax, rMin, frame)
    return circles