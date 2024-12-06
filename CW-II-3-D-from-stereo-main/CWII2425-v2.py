'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    groundTruth = []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(int(args.sph_sep_min),int(args.sph_sep_max),1)
        x = random.randrange(int(-h/2+2), int(h/2-2), step)
        z = random.randrange(int(-w/2+2), int(w/2-2), step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(int(-h/2+2), int(h/2-2), step)
            z = random.randrange(int(-w/2+2), int(w/2-2), step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)
        groundTruth.append([x, size, z])

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]

#####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam,True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    
    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################

    #########################################################################################################
    #Functions
    def getColour(index):
    # Calculate RGB values by cycling through the channels
        r = int((index * 30) % 256)  # Red channel (cycling every 256 colors)
        g = int((index * 60) % 256)  # Green channel (cycling every 256 colors)
        b = int((index * 90) % 256)  # Blue channel (cycling every 256 colors)
        return (b, g, r)  # OpenCV uses BGR, not RGB, so we return (B, G, R)
    

    #########################################################################################################
    #Task 3

    img0 = cv2.imread('view0.png', 1)
    img1 = cv2.imread('view1.png', 1)
    img_gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    img_gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    blur0 = cv2.medianBlur(img_gray0, 5)
    blur1 = cv2.medianBlur(img_gray1, 5)

    circles0 = cv2.HoughCircles(blur0,cv2.HOUGH_GRADIENT,1.5,15, param1=100, param2=35, minRadius=10, maxRadius=45)
    circles0 = np.uint16(np.around(circles0))
    for i in circles0[0,:]:
        cv2.circle(img0,(i[0],i[1]),i[2],(0,255,0),2) # draw the circle
        cv2.circle(img0,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
    cv2.imwrite('circles0.png', img0) # save image


    circles1 = cv2.HoughCircles(blur1,cv2.HOUGH_GRADIENT,1.5,15, param1=100, param2=35, minRadius=10, maxRadius=45)
    circles1 = np.uint16(np.around(circles1))
    index = 1
    colours = []
    for i in circles1[0,:]:
        colour = getColour(index)
        colours.append(colour)
        cv2.circle(img1,(i[0],i[1]),i[2],colour,2) # draw the circle
        cv2.circle(img1,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
        index += 1
    cv2.imwrite('circles1.png', img1) # save image


    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here

    Notes:
    Get Rotation and Translation for both cameras seperated (from the 4x4 matrix into 3x3 and 1x3)
    Calculate the rotation and translation from cam0 to cam1
    Once you have T construct the matrix S (Skew)
    Then get the instrinsic matrix from the code above (M)
    Calculate the Essential Matrix RS
    Calculate M Transposed = MT
    Calculate the Fundermental Matrix F = MT * E * M
    Calculate the epipolar lines for each centre Line = F * C where C = [x,y,1]
    Line is of the form ax + by + c = 0
    Calculate the intersection of our image (Furthest left and right points)
    Plot onto the second image
    '''
    ###################################
    R0 = H0_wc[:3, :3] #rotation matrix from world to cam0
    R1 = H1_wc[:3, :3] #rotation matrix from world to cam1
    T0 = H0_wc[:3, 3] #translation matrix from world to cam0
    T1 = H1_wc[:3, 3] #translation matrix from world to cam0
    R = R1 @ R0.T #rotation matrix from cam0 to cam1
    T = T1 - R@T0 #translation matrix from cam0 to cam1
    S = np.array([[0, -T[2], T[1]], 
                  [T[2], 0, -T[0]],
                  [-T[1], T[0], 0]])
    E = S @ R #essential matrix
    M = K.intrinsic_matrix 
    MInv = np.linalg.inv(M)
    F = MInv.T @ E @ MInv #Fundermental Matrix

    lines = []
    points = []
    centres0 = []
    for centre in circles0[0,:]:
        centres0.append(centre)
        C = np.array([centre[0], centre[1], 1])
        line = F @ C
        lines.append(line)

        a, b, c = line
        x0 = 0
        y0 = int(-c / b)
        x1 = img_width
        y1 =  int(-(c + a*x1) / b)

        points.append([(x0, y0), (x1, y1)])
        cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Show the image with epipolar lines
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines")
    plt.show()


    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################

    #########################################################################################################
    #Functions
    def distanceToLine(line, centre):
        a,b,c = line
        x, y = centre
        return abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    

    #########################################################################################################
    #Task 5

    closest = []
    correspondingCentres = [] #[centre0, centre1]
    for i in range (len(lines)):
        line = lines[i]
        minDistance = float('inf')
        currClosest = None
        index = 0
        for centre in circles1[0,:]:
            distance = distanceToLine(line, [centre[0], centre[1]])
            if distance < minDistance and distance < 15:
                minDistance = distance
                currClosest = [centre, colours[index]]
            index += 1
        if currClosest != None:
            correspondingCentres.append([[centres0[i][0], centres0[i][1], 1], [currClosest[0][0], currClosest[0][1], 1], centres0[i][2]])
            cv2.line(img1, points[i][0], points[i][1], currClosest[1], 2)

    # Show the image with epipolar lines
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title("Epipolar Lines with colour")
    plt.show()


    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here

    Notes:
    Pcam0 = R.T @ Pcam1 + T
    Pcam1 = R(Pcam0 - T)
    Going to work in cam0
    (a * Pcam0) - (b * (R.T @ Pcam1) - t) - c((Pcam0)  CrossProduct with -> (R.T @ Pcam1)) = 0
    R = rotation from cam0 to cam1
    t = translation from cam0 to cam1
    .T = transpose
    a, b, c are all scalars
    We can then find a,b,c by rearranging the equation to give us:
    [a, b, c].T = HInv @ T
    Where HInv is the inverse of H
    H is some matrix from all the parts in the first equation
    '''
    ###################################
    myCentres = []
    worldCentres = []
    for centres in correspondingCentres:
        ###################################
        #1st Attempt
        centre0 = np.array(centres[0])
        centre1 = np.array(centres[1])
        centre1ToCam0 = np.array((R.T @ centre1) - T )

        term1 = centre0
        term2 = -centre1ToCam0
        term3 = -np.cross(centre0, (R.T @ centre1))

        H = np.column_stack((term1, term2, term3))
        InvH = np.linalg.inv(H)
        a, b, c = InvH @ T
        point = ((a * centre0) + (b * (R.T @ centre1)) + T) / 2

        worldPoint = R0.T @ point + T0
        myCentres.append(worldPoint)        


        ###################################
        #Working Attempt
        camLCentre = -R1.T @ T1
        camRCentre = -R0.T @ T0

        PL = MInv @ np.array([centres[1][0],centres[1][1],1])
        PR = MInv @ np.array([centres[0][0],centres[0][1],1])

        PLWorld = R1.T @ PL
        PRWorld = R0.T @ PR

        H = np.column_stack([PLWorld, -PRWorld, -np.cross(PLWorld, PRWorld)])

        TWorld = camRCentre - camLCentre
    
        HInv = np.linalg.inv(H)

        a,b,c = HInv @ TWorld.reshape(-1, 1)
        P = ((camLCentre + a * PLWorld) + (camRCentre + b * PRWorld)) / 2
        worldCentres.append([P[0], P[1], P[2]])


    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################

    #########################################################################################################
    #Functions
    def creatingCentres(worldCentres):
        Mesh_list = []
        RGB_list = []
        H_list = []
        for centres in worldCentres:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            Mesh_list.append(sphere)
            RGB_list.append([0.0, 0.0, 1.0])  # Use red for calculated spheres

            # Create transformation matrix for the sphere's position
            sph_H = np.array(
                [[1, 0, 0, centres[0]],  # X position
                [0, 1, 0, centres[1]],  # Y position
                [0, 0, 1, centres[2]],  # Z position
                [0, 0, 0, 1]]
            )
            H_list.append(sph_H)

        Obj_meshes = []
        for (mesh, H, rgb) in zip(Mesh_list, H_list, RGB_list):
            # Apply location transformation
            mesh.vertices = o3d.utility.Vector3dVector(
                transform_points(np.asarray(mesh.vertices), H)
            )
            # Paint meshes with uniform colors
            mesh.paint_uniform_color(rgb)
            mesh.compute_vertex_normals()
            Obj_meshes.append(mesh) 
        return Obj_meshes
    

    ###################################
    def display(Obj_meshes, args, obj_meshes):
        pcd_GTcents = o3d.geometry.PointCloud()
        pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
        pcd_GTcents.paint_uniform_color([1., 0., 0.])
        if args.bCentre:
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=640, height=480, left=0, top=0)
            for i in range(len(Obj_meshes)):
                for m in [Obj_meshes[i], pcd_GTcents]:
                    vis.add_geometry(m)
                for m in [obj_meshes[0], pcd_GTcents]:
                    vis.add_geometry(m)
            vis.run()
            vis.destroy_window()


    ###################################
    def errors(worldCentres):
        errors = []
        for centre in worldCentres:
            minErr = float('inf')
            for real in groundTruth:
                error = np.sqrt((real[0] - centre[0])**2 + (real[1] - centre[1])**2 + (real[2] - centre[2])**2)
                if error < minErr:
                    minErr = error

            errors.append(minErr)
        errorResults(errors)


    ###################################
    def errorResults(errors):
        meanError = 0
        RMSE = 0
        for error in errors:
            meanError += error
            RMSE += error**2
        meanError = meanError / len(errors)
        RMSE = np.sqrt(RMSE / len(errors))
        maxError = max(errors)

        print("Errors: ", errors)
        print("Mean Error: ", meanError)
        print("Root Mean Squared Error: ", RMSE)
        print("Max Error: ", maxError)
        print("")


    #########################################################################################################
    #Task 7
    
    #Working Centres
    Obj_meshes = creatingCentres(worldCentres)
    display(Obj_meshes, args, obj_meshes)
    print("Errors of Centre Estimates")
    errors(worldCentres)

    #Initial Attempt
    # Obj_meshes = creatingCentres(myCentres)
    # display(Obj_meshes, args, obj_meshes)
    # errors(myCentres)



    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################
    radii = []
    for i in range(len(correspondingCentres)):
        centre0 = correspondingCentres[i][0]
        worldCentre = worldCentres[i]
        distance = np.sqrt((worldCentre[0] - T0[0])**2 + (worldCentre[1] - T0[1])**2 + (worldCentre[2] - T0[2])**2)
        radius = (correspondingCentres[i][2] * distance) / f
        radii.append(radius)


    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################

    #########################################################################################################
    #Functions
    def creatingSphereFrames(worldCentres, radii, colour):
        Mesh_list = []
        RGB_list = []
        H_list = []
        for i in range(len(worldCentres)):
            centres = worldCentres[i]
            r = radii[i]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            
            vertices = np.asarray(sphere.vertices)
            triangles = np.asarray(sphere.triangles)
            
            
            edges = []
            for triangle in triangles: # Create a list of edges (lines) from the triangles
                edges.append([triangle[0], triangle[1]])
                edges.append([triangle[1], triangle[2]])
                edges.append([triangle[2], triangle[0]])

            # Convert the edges into a LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            
            frameSphere = line_set.translate(centres)

            Mesh_list.append(frameSphere)
            RGB_list.append(colour) 
            H_list.append(np.eye(4))

        Obj_meshes = []
        for (mesh, H, rgb) in zip(Mesh_list, H_list, RGB_list):
            points = np.asarray(mesh.points)
            transformed_points = transform_points(points, H)
            mesh.points = o3d.utility.Vector3dVector(transformed_points)
            mesh.paint_uniform_color(rgb)
            Obj_meshes.append(mesh)
        return Obj_meshes
    

#########################################################################################################
    #Task 9

    realRadii = []
    for centre in groundTruth:
        realRadii.append(centre[1])

    Obj_meshes = creatingSphereFrames(worldCentres, radii, [0.0, 0.0, 1.0])
    Obj_meshes += (creatingSphereFrames(groundTruth, realRadii, [1.0, 0.0, 0.0]))
    display(Obj_meshes, args, obj_meshes)

    errors = []
    meanError = 0
    RMSE = 0
    for i in range(len(radii)):
        errors.append(radii[i]-realRadii[i])

    print("Errors of Radii Estimates")
    errorResults(errors)

    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
