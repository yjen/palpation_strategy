import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.io import savemat

LEFT = '/scene1.001.png'
RIGHT = '/scene1.002.png'
try:
    stereo = cv2.StereoSGBM(0, 256, 11, P1=100, P2=200, disp12MaxDiff=181, preFilterCap=4,\
        uniquenessRatio=5, speckleWindowSize=150, speckleRange=16, fullDP=True)
except AttributeError:
    stereo = cv2.StereoSGBM_create(0, 256, 11, P1=100, P2=200, disp12MaxDiff=181,
                        preFilterCap=4, uniquenessRatio=1,
                        speckleWindowSize=150, speckleRange=16)

                                   
folders = [x[0] for x in os.walk('image_pairs')]
IMAGE_SIZE = 500


# 3x3 Projective Transformation
def computeH(im1_pts, im2_pts):
    # inverse transforming im2_pts to im1_pts
    # the transformation is a homography: p'=Hp, where H is a 3x3 matrix with 8 degrees of freedom (lower right corner is a scaling factor and can be set to 1)
    # im1_pts and im2_pts are n-by-2 matrices holding the (x,y) locations of n point correspondences from the two images
    if im1_pts.shape != im2_pts.shape or len(im1_pts.shape) != 2:
        print "Different number of correspondences!"
        sys.exit(1)
        return
    num_pts = im1_pts.shape[0]
    if num_pts < 4:
        print "Too few correspondences, need at least 4 pairs of points"
        sys.exit(1)
        return

    # computing A*h = b, solving for h, where h = [a,b,c,d,e,f,g,h].T
    A = np.zeros((num_pts*2, 8))
    b = np.zeros((num_pts*2, 1))
    for i in range(num_pts):
        new_x, new_y = im1_pts[i]
        x, y = im2_pts[i]
        A[i*2, :3] = x, y, 1
        A[i*2, 6:] = -x*new_x, -y*new_x
        A[i*2+1, 3:] = x, y, 1, -x*new_y, -y*new_y
        b[i*2:i*2+2] = np.array([new_x, new_y]).reshape((2,1))

    h = np.linalg.lstsq(A, b)[0]
    h = np.append(h, [1])
    H = h.reshape((3,3))

    return H
def interp_function(image):
    # creating interpolation functions
    x = np.array(range(image.shape[0]))
    y = np.array(range(image.shape[1]))
    # R,G,B interpolators
    return RGI((x, y), image, bounds_error=False, fill_value=0)

# Crops the center plane we want, rectifies as if we are looking at it from above
def rectify(image):
        im1_pts = np.array([[210, 630], [300, 210], [770, 210], [870, 630]])
        im2_pts = np.array([[0, 499], [0, 0], [499, 0], [499, 499]])

        H = computeH(im1_pts, im2_pts)

        rectified = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        rr, cc = np.where(rectified == 0)
        indices = np.array([cc, rr, np.ones(len(rr))]).reshape((3, len(rr)))
        indices_to_obtain = np.dot(H, indices)
        indices_to_obtain[0] /= indices_to_obtain[2]
        indices_to_obtain[1] /= indices_to_obtain[2]
        indices_to_obtain = np.array([indices_to_obtain[1], indices_to_obtain[0]])
        interpolator = interp_function(image)
        rectified = interpolator(indices_to_obtain.T).reshape(rectified.shape)
        return rectified


def getDisparityMap(folder):
    left = cv2.imread(folder + LEFT, 1)
    right = cv2.imread(folder + RIGHT, 1)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    disparity = stereo.compute(left, right)
    return disparity


def getDefaultBelief():
    belief = np.ones((IMAGE_SIZE, IMAGE_SIZE))
    a = (np.ones(IMAGE_SIZE)*0.997)**range(IMAGE_SIZE)
    return (belief*a).T[::-1]




for i, folder in enumerate(folders[1:]):
    disparity = getDisparityMap(folder)/16
    # print(disparity.dtype)
    # c_x = len(disparity[0])/2
    # c_xp = c_x
    # c_y = len(disparity)/2
    # f = 35
    # T_x = 14
    # Q = np.array([[1, 0, 0, -c_x], [0, 1, 0, -c_y], [0, 0, 0, f], [0, 0, -1.0/T_x, (c_x-c_xp)/T_x]])
    # threeD = cv2.reprojectImageTo3D(disparity, Q)
    # print(threeD)
    # plt.imsave(folder+ "/disparity.mat", disparity)
    
    # plt.imsave(folder+ "/rectified.mat", rectified)
    # belief = getDefaultBelief()
    # plt.imsave(folder+ "/belief.png", np.dstack([belief, belief, belief]))




    print("Max disp: " + str(np.max(disparity)))
    print("Min disp: " + str(np.min(disparity)))
    # print("Changed zeros to 1e10")
    print(disparity[200:600, 350:750])
    disparity2 = disparity.astype(np.float32) #+ 17
    disparity2[disparity2 < 10] = 35
    print(disparity2[200:600, 350:750])
    print("Max disp: " + str(np.max(disparity2)))
    print("Min disp: " + str(np.min(disparity2)))

    # for i in range(len(disparity2)):
    #     for j in range(len(disparity2[0])):
    #         if disparity2[i][j] <= 0:
    #             disparity2[i][j] = 1
    rectified = rectify(disparity2)
    # rectified[rectified < 200] = 600
    print("max rect: " + str(rectified.max()))
    print("min rect: " + str(rectified.min()))
    # plt.hist(rectified.flatten())
    # plt.show()
    print(np.sum(rectified[0])/500)
    print(np.sum(rectified[499]/500))

    # print("Max disp2: " + str(np.max(disparity2)))
    # print("Min disp2: " + str(np.min(disparity2)))
    FT = 1200*20#*(35/2) #995.603 #1200 more closely matches geometry
    depth = FT / rectified
    print("depth first row avg: " + str(np.sum(depth[0])/500))
    print("max depth first row: " + str(np.max(depth[0])))
    print("min depth first row: " + str(np.min(depth[0])))
    print("max depth last row: " + str(np.max(depth[499])))
    print("min depth last row: " + str(np.min(depth[499])))
    print("depth last row avg: " + str(np.sum(depth[499])/500))

    depth_display = depth.copy()
    depth_display[depth_display > 1] = 1 #1

    # print("Max depth: " + str(np.max(depth)))
    # print("Min depth: " + str(np.min(depth)))
    # plt.imsave(folder + "/depth.mat", depth)
    rect_depth = rectify(depth)
    rect_depth_display = rectify(depth_display)
    rect_depth_display[rect_depth_display > 1] = 1
    rect_depth_display[rect_depth_display > 0.7] = 0.7
    rect_depth_display[rect_depth_display < 0.45] = 0.45

    rect_depth_display = depth
    # plt.imsave(folder + "/rectified_depth.mat", rectified_depth)
    
    # print("disparity")
    # print(disparity)
    # print('disparity2')
    # print(disparity2)
    # print('depth')
    # print(depth)
    # print('rectified')
    # print(rectified)
    # print('min rect: ' + str(np.min(rectified)))
    # print('max rect: ' + str(np.max(rectified)))
    # print('rectified_depth')
    # print(rect_depth)
    # print('min rectd: ' + str(np.min(rect_depth)))
    # print('max rectd: ' + str(np.max(rect_depth)))
    # print("disparity2 shape: " + str(disparity2.shape))
    # print("rectified shape: " + str(rectified.shape)) 
    # print("depth shape: " + str(depth.shape))
    # print("rect_d shape: " + str(rect_depth.shape))
    # print('min rectdd: ' + str(np.min(rect_depth_display)))
    # print('max rectdd: ' + str(np.max(rect_depth_display)))
    # print(rect_depth_display)
    # print(np.sum(rect_depth_display[0]))
    # print(np.sum(rect_depth_display[499]))
    a1 = np.ones((500,500))
    for n in range(500):
        a1[n] = 250 + (499-n)*200.0/499
    # print('a1: ' + str(a1))
    a3 = 300
    theta2 = np.arctan(a1/a3)
    # print('theta2: ' + str(theta2))
    theta1 = np.pi/2 - theta2
    # print('theta1: ' + str(theta1))
    x1 = rect_depth_display * np.sin(theta1)
    x1 = 300 - x1
    # x1[x1 < 25] = 25 (for F=995.603)
    x1[x1 < -20] = -20 # for (F=1200). actually doesn't work that well
    print('max x1: ' + str(np.max(x1)))
    print('min x1: ' + str(np.min(x1)))
    # plt.hist(x1.flatten())
    # plt.show()
    x2 = a1 * np.tan(theta1) #okay this makes sense. x1&x3 are different but it's okay I think.
    # print('a1: ' + str(a1))
    # print('theta1: ' + str(theta1))
    # print('rectdepthdisplay: ' + str(rect_depth_display))
    # print('rectifieddisp: ' + str(rectified))  
    # print(np.square(rect_depth_display) - np.square(a1))
    x3 = np.sqrt(np.square(rect_depth_display) - np.square(a1))

    colormap = 'seismic'
    plt.figure(figsize = (20,8))
    plots = [rect_depth_display, x1, x2, x3]#disparity2, rectified, depth_display, rect_depth_display, x1, x2, x3]
    numplots = len(plots)
    start = 101 + numplots*10
    
    for i in range(numplots):
        plt.subplot(start+i)
        plt.imshow(plots[i], cmap=colormap)
    
    print('x1: ' + str(x1))
    print('x2: ' + str(x2))
    print('x3: ' + str(x3))
    # print('rd: ' + str(rect_depth))
    # plt.show()
    # let's do gradients w/ steps of 10

    x1_smoothed = cv2.GaussianBlur(x1, (43,43), 0)

    h = 5
    x1_padded = np.zeros((500+2*h,500+2*h))
    x1_padded[h:len(x1_padded)-h, h:len(x1_padded)-h] = x1_smoothed
    xgrad = np.zeros((500,500))
    ygrad = np.zeros((500,500))


    for i in range(500):
        for j in range(500):
            ygrad[i][j] = (x1_padded[i+h+h][j+h] - x1_padded[i-h+h][j+h])/(2*h)
            xgrad[i][j] = (x1_padded[i+h][j+h+h] - x1_padded[i+h][j-h+h])/(2*h)
    print("xgrad: " + str(xgrad))
    print("ygrad: " + str(ygrad))
    print("xgrad shape: " + str(xgrad.shape))
    print("ygrad shape: " + str(ygrad.shape))
    print("x1: " + str(x1))
    print("x1_smoothed: " + str(x1_smoothed))
    # plots = []
    # for i in range(10):
    #     plots.append(cv2.GaussianBlur(x1, (10*i + 3,10*i + 3), 0))
    
    # for j in range(len(plots)):
    #     plt.subplot(2, 5, j+1)
    #     plt.imshow(plots[j], cmap=colormap)
    # plt.show()

    plt.subplot(141)
    plt.imshow(xgrad, cmap=colormap)
    plt.subplot(142)
    plt.imshow(ygrad, cmap=colormap)
    plt.subplot(143)
    plt.imshow(x1, cmap=colormap)
    plt.subplot(144)
    plt.imshow(x1_smoothed, cmap=colormap)
    plt.show()
    # print("x1")
    # for i in range(len(x1)):
    #     print("i: " + str(x1[i]))
    #     a=raw_input("blahwtfisthis: ")
    savemat('depth_grad_maps.mat', {'depth':x1, 'ygrad':ygrad, 'xgrad':xgrad})
