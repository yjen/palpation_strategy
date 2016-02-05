import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.io import savemat
from observations import _observationModel

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
IMG_SIZE = 500

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

# Crops the center plane we want, reshapes into IMG_SIZExIMG_SIZE matrix
def project(image):
        im1_pts = np.array([[210, 630], [300, 210], [770, 210], [870, 630]])
        im2_pts = np.array([[0, 499], [0, 0], [499, 0], [499, 499]])

        H = computeH(im1_pts, im2_pts)

        projection = np.zeros((IMG_SIZE, IMG_SIZE))
        rr, cc = np.where(projection == 0)
        indices = np.array([cc, rr, np.ones(len(rr))]).reshape((3, len(rr)))
        indices_to_obtain = np.dot(H, indices)
        indices_to_obtain[0] /= indices_to_obtain[2]
        indices_to_obtain[1] /= indices_to_obtain[2]
        indices_to_obtain = np.array([indices_to_obtain[1], indices_to_obtain[0]])
        interpolator = interp_function(image)
        projection = interpolator(indices_to_obtain.T).reshape(projection.shape)
        return projection

def getDisparityMap(folder):
    left = cv2.imread(folder + LEFT, 1)
    right = cv2.imread(folder + RIGHT, 1)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
    disparity = stereo.compute(left, right)
    return disparity

def getDefaultBelief():
    belief = np.ones((IMG_SIZE, IMG_SIZE))
    a = (np.ones(IMG_SIZE)*0.997)**range(IMG_SIZE)
    return (belief*a).T[::-1]

def compute_depth(F, T, disparity):
    return F*T/disparity

def compute_topdown_height(depth, size, x, y):
    a1 = np.ones((IMG_SIZE,IMG_SIZE))
    for n in range(IMG_SIZE):
        a1[n] = x + (IMG_SIZE-1-n)*size/(IMG_SIZE-1)
    a3 = y
    theta2 = np.arctan(a1/a3)
    theta1 = np.pi/2 - theta2
    x1 = depth * np.sin(theta1)
    x1 = y - x1
    return x1

def getObservationModel(planeName):
    model = np.array(_observationModel[planeName])
    # print("model: " + str(model))
    # print("shape: " + str(len(model)) + " " + str(len(model[0])))
    if model is None:
        return None
    # model is 21x21 points, we want the image to be 500x500
    obs = cv2.resize(model, (IMG_SIZE,IMG_SIZE))#, interpolation=cv2.INTER_LANCZOS4)
    # print("obs: " + str(obs))
    # print("shape: " + str(obs.shape))
    return obs

# main loop
for i, folder in enumerate(folders[1:]):
    if folder == 'image_pairs/exp' or folder == 'image_pairs/halved' or folder == 'image_pairs/plain_random' or folder == 'image_pairs/squaredDiffs':
        continue
    print("folder: " + str(folder))
    disparity = getDisparityMap(folder).astype(np.float32)/16
    print(disparity.dtype)
    
    # plt.imsave(folder+ "/disparity.mat", disparity)
    # plt.imsave(folder+ "/rectified.mat", rectified) #rectified is wrong word?
    # belief = getDefaultBelief()
    # plt.imsave(folder+ "/belief.png", np.dstack([belief, belief, belief]))


    projected_disp = project(disparity)
    projected_disp = projected_disp.astype(np.float32)
    print(projected_disp.dtype)
    projected_disp_new = cv2.medianBlur(projected_disp, 5)
    print("diff: " + str(np.sum(np.abs(projected_disp_new - projected_disp))))
    projected_disp = projected_disp_new
    # plt.hist(projected_disp.flatten())
    # values = sorted(projected_disp.flatten())
    # print("1st percentile: " + str(values[int(0.01*len(values))]))
    # plt.show()

    values = sorted(projected_disp.flatten())
    floor = values[int(0.01*len(values))]
    projected_disp[projected_disp < floor] = floor
    ceil = values[int(0.99*len(values))]
    projected_disp[projected_disp > ceil] = ceil
    projected_disp[projected_disp < 0] = 0

    avg_disp = np.sum(projected_disp)/np.square(IMG_SIZE)
    projected_disp[projected_disp < 1] = avg_disp
    print("min disp: " + str(np.min(projected_disp)))
    print("max disp: " + str(np.max(projected_disp)))
    # plt.hist(projected_disp.flatten())
    # plt.show()

    F = 1200 #995.603 is the value we get from the equation, but 120 more closely matches geometric model & ground truth
    T = 20
    depth = compute_depth(F, T, projected_disp)
    print("min depth: " + str(np.min(depth)))
    print("max depth: " + str(np.max(depth)))
    # plt.hist(depth.flatten())
    # plt.show()
    # plt.imsave(folder + "/depth.mat", depth)

    size = 200.0
    x = 250
    y = 300
    x1 = compute_topdown_height(depth, size, x, y)
    values = sorted(x1.flatten())
    floor = values[int(0.01*len(values))]
    ceil = values[int(0.99*len(values))]
    x1[x1 < floor] = floor
    x1[x1 > ceil] = ceil
    # print(floor)
    # plt.hist(x1.flatten())
    # plt.show()
    print("a")
    print(x1.dtype)
    print("b")
    x1_median = x1.astype(np.float32)
    num_iter1 = 5
    num_iter2 = 25
    kernel_size = 5
    for _ in range(num_iter1):
    	x1_median = cv2.medianBlur(x1_median,kernel_size)
    x1_median2 = x1_median.astype(np.float32)
    for _ in range(num_iter2-num_iter1):
    	x1_median2 = cv2.medianBlur(x1_median2,kernel_size)



    colormap = 'Blues'
    plt.figure(figsize = (20,8))

    g_kernel_size1 = 43
    g_kernel_size2 = 25
    x1_smoothed = cv2.GaussianBlur(x1_median2, (g_kernel_size1, g_kernel_size1), 0)
    x1_smoothed2 = cv2.GaussianBlur(x1_median2, (g_kernel_size2, g_kernel_size2), 0)
    # x1_smoothed[:470, 30:] = x1_smoothed[30:, :470] #OMG the problem is that the "ground truth" we're reading is rotated by 180*...

    
    avg_x1 = np.sum(x1_smoothed)/np.square(IMG_SIZE)
    print("avg_x1: " + str(avg_x1))
    obs = getObservationModel(folder)*10 #convert to mm
    if folder == 'image_pairs/sinxy1':
    	obs = np.rot90(obs,1) #correct for sinxy1, wrong for the smoothed images
    else:
    	obs = np.fliplr(obs)
    avg_obs = np.sum(obs)/np.square(IMG_SIZE)
    print("avg_obs: " + str(avg_obs))
    x1_smoothed = x1_smoothed - (avg_x1-avg_obs) # this is where I center the prediction around the obs...sketch LOL.
    print("new avg_x1: " + str(np.sum(x1_smoothed)/np.square(IMG_SIZE)))


    h = 5
    x1_padded = np.zeros((IMG_SIZE+2*h,IMG_SIZE+2*h))
    x1_padded[h:len(x1_padded)-h, h:len(x1_padded)-h] = x1_smoothed
    xgrad = np.zeros((IMG_SIZE,IMG_SIZE))
    ygrad = np.zeros((IMG_SIZE,IMG_SIZE))

    for i in range(IMG_SIZE):
        for j in range(IMG_SIZE):
            ygrad[i][j] = (x1_padded[i+h+h][j+h] - x1_padded[i-h+h][j+h])/(2*h)
            xgrad[i][j] = (x1_padded[i+h][j+h+h] - x1_padded[i+h][j-h+h])/(2*h)

    grad = np.sqrt(np.square(xgrad) + np.square(ygrad))
    
    errors = np.abs(obs - x1_smoothed)
    sse = np.sum(np.square(obs - x1_smoothed))
    mae = np.sum(np.abs(obs - x1_smoothed))/np.square(IMG_SIZE)
    print(obs)
    print(x1_smoothed)
    print("SSE: " + str(sse))
    print("MSE: " + str(sse/np.square(IMG_SIZE)))
    print("mean absolute error: " + str(mae))
    # for finding appropriate size of gaussian kernel
    # plots = []
    # for i in range(10):
    #     plots.append(cv2.GaussianBlur(x1, (10*i + 3,10*i + 3), 0))
    
    # for j in range(len(plots)):
    #     plt.subplot(2, 5, j+1)
    #     plt.imshow(plots[j], cmap=colormap)
    # plt.show()

    # obs = getObservationModel(folder)
    # print("folder: " + str(folder))
    # print("obs: " + str(obs))
    # print("obs type: " + str(obs.shape))


    print("min disp: " + str(np.min(projected_disp)))
    print("max disp: " + str(np.max(projected_disp)))

    left = cv2.imread(folder + LEFT, 1)
    right = cv2.imread(folder + RIGHT, 1)

    a1=plt.subplot(3,4,1)
    a1.set_title("Right image")
    plt.imshow(right)
    plt.colorbar()
    a2=plt.subplot(3,4,2)
    a2.set_title("Projected disparity")
    plt.imshow(projected_disp, cmap=colormap)
    plt.colorbar()
    a3=plt.subplot(3,4,3)
    a3.set_title("Projected depth map")
    plt.imshow(depth, cmap=colormap)
    plt.colorbar()
    a4=plt.subplot(3,4,4)
    a4.set_title("Topdown height map")
    plt.imshow(x1, cmap=colormap)
    plt.colorbar()

    a5=plt.subplot(3,4,8)
    a5.set_title("Median filtered topdown height map " + str(num_iter2))
    plt.imshow(x1_median2, cmap=colormap)
    plt.colorbar()
    a6=plt.subplot(3,4,7)
    a6.set_title("Gaussian smoothed topdown height map ")
    plt.imshow(x1_smoothed, cmap=colormap)
    plt.colorbar()
    a7=plt.subplot(3,4,6)
    a7.set_title("Gaussian smoothed topdown height map ")
    plt.imshow(x1_smoothed, cmap=colormap)
    plt.colorbar()

    a8=plt.subplot(3,4,5)
    a8.set_title("Gradient norms")
    plt.imshow(grad, cmap=colormap)
    plt.colorbar()
    a9=plt.subplot(3,4,9)
    a9.set_title("Errors")
    plt.imshow(errors, cmap=colormap)
    plt.colorbar()
    a10=plt.subplot(3,4,10)
    a10.set_title("Ground truth")
    plt.imshow(obs, cmap=colormap)
    plt.colorbar()


    plt.show()
    output_name = folder[(len('image_pairs/')):] + '_depth_grad_maps.mat'
    savemat(output_name, {'height':x1_smoothed, 'ygrad':ygrad, 'xgrad':xgrad})

