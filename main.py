import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans

test_image_set_idx = 2
cluster_size = 10
T_reduce = 50
T_ini = 0.01
sam_dist = 3
morph_size =11


## camera instrinsic param
fx = 544.2582548211519  # focal length x
fy = 546.0878823951958  # focal length y
cx = 326.8604521819424  # optical center x
cy = 236.1210149172594  # optical center y
k1 = 0.0369 #distortion coefficients
k2 = -0.0557 #distortion coefficients 2

def Depth_preprocess(raw_depth_image):
    for i in range(len(raw_depth_image)):
        raw_depth_image[i] = raw_depth_image[i].astype(np.float64)/1000#depth image(meter)

def pixel2Point3D(i,j,z):
    X = (j-cx)*z/fx
    Y = (i-cy) * z / fy
    Z =z
    return [X, Y, Z]

#read image and depth image
print("image idx ", test_image_set_idx)
depth_names = glob.glob("depth/test" + str(test_image_set_idx) + "/*.png")
image_names = glob.glob("image/test" + str(test_image_set_idx) + "/*.png")
dmaps = [cv2.imread(depth_names[i],cv2.IMREAD_ANYDEPTH) for i in range(len(depth_names))]
images = [cv2.imread(image_names[i]) for i in range(len(image_names))]

## depth image morphology
for i in range(2):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    dmaps[i] = cv2.morphologyEx(dmaps[i],cv2.MORPH_CLOSE,k)

Depth_preprocess(dmaps)
for i, distorted_image in enumerate(images):
    camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    undist_image = cv2.undistort(distorted_image,camera_matrix,np.array([k1,k2,0,0]))
    images[i] = undist_image


### change pixels to 3d points
depth_points = []
depth_pixel = []
image_height, image_width, _ = images[0].shape
for i in range(image_height):
    for j in range(image_width):
        if dmaps[0][i,j] !=0:
            depth_points.append(pixel2Point3D(i,j,dmaps[0][i,j]))
            depth_pixel.append([i,j])

### k means clustering of 3d points
kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(np.array(depth_points))
result_image = np.zeros((image_height,image_width,3),dtype=np.uint8)
cluster_idx_image = np.full((image_height,image_width),-1)
random_colors = []
for _ in range(cluster_size):
    random_colors.append([int(j) for j in np.random.randint(0,255, 3)])
for i,point in enumerate(depth_pixel):
    color_idx = kmeans.labels_[i]
    result_image[point[0],point[1]] = random_colors[color_idx]
    cluster_idx_image[point[0],point[1]] = color_idx

### extract Orb feature
source = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
target = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(
    nfeatures=10000
)
kp_source, des_source = orb.detectAndCompute(source,None)
kp_target, des_target = orb.detectAndCompute(target,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

### match feature
matches = bf.match(des_source,des_target)
matches = sorted(matches, key = lambda x:x.distance)
pts_source = []
pts_target = []
for k, match in enumerate(matches):
    pts_source.append(kp_source[match.queryIdx].pt)
    pts_target.append(kp_target[match.trainIdx].pt)
pts_source_list = pts_source.copy()
pts_target_list = pts_target.copy()
pts_source = np.float64(pts_source)
pts_target = np.float64(pts_target)

### find fundamental matrix
F, mask = cv2.findFundamentalMat(pts_source,pts_target,cv2.FM_LMEDS)
feature_image = result_image.copy()

for point in pts_source_list:
    feature_image = cv2.circle(feature_image,(int(point[0]),int(point[1])),3,(0,0,255),-1)

inlier_image = result_image.copy()
inlier_pts_source_list = []
inlier_pts_target_list = []
N_ini = [0 for _ in range(cluster_size)]
N_first = [0 for _ in range(cluster_size)]
N_second = [0 for _ in range(cluster_size)]

### find first inlier
for i in range(len(pts_source_list)):
    dist = cv2.sampsonDistance(np.float64([pts_source[i][0],pts_source[i][1],1]),np.float64([pts_target[i][0],pts_target[i][1],1]),F)
    pt_x = int(pts_source[i][0])
    pt_y = int(pts_source[i][1])
    if cluster_idx_image[pt_y,pt_x] != -1:
        N_ini[cluster_idx_image[pt_y,pt_x]] = N_ini[cluster_idx_image[pt_y,pt_x]] + 1

    if dist <sam_dist:
        inlier_image = cv2.circle(inlier_image,(int(pts_source[i][0]),int(pts_source[i][1])),3,(0,0,255),-1)
        inlier_pts_source_list.append([pts_source[i][0],pts_source[i][1]])
        inlier_pts_target_list.append([pts_target[i][0], pts_target[i][1]])
        if cluster_idx_image[pt_y, pt_x] != -1:
            N_first[cluster_idx_image[pt_y, pt_x]] = N_first[cluster_idx_image[pt_y, pt_x]] + 1
    else:
        inlier_image = cv2.circle(inlier_image, (int(pts_source[i][0]), int(pts_source[i][1])), 3, (255, 0, 0), -1)

inlier_pts_source = np.float64(inlier_pts_source_list)
inlier_pts_target = np.float64(inlier_pts_target_list)
inlier_image_second = result_image.copy()

### find second fundamental matrix
F_inlier, mask = cv2.findFundamentalMat(inlier_pts_source,inlier_pts_target,cv2.FM_LMEDS)

### find second inlier
for i in range(len(inlier_pts_source_list)):
    dist = cv2.sampsonDistance(np.float64([inlier_pts_source[i][0],inlier_pts_source[i][1],1]),
                               np.float64([inlier_pts_target[i][0],inlier_pts_target[i][1],1]),F_inlier)
    pt_x = int(inlier_pts_source[i][0])
    pt_y = int(inlier_pts_source[i][1])
    if dist <sam_dist:
        inlier_image_second = cv2.circle(inlier_image_second,(int(inlier_pts_source[i][0]),int(inlier_pts_source[i][1])),3,(0,0,255),-1)
        if cluster_idx_image[pt_y, pt_x] != -1:
            N_second[cluster_idx_image[pt_y, pt_x]] = N_second[cluster_idx_image[pt_y, pt_x]] + 1
    else:
        inlier_image_second = cv2.circle(inlier_image_second, (int(inlier_pts_source[i][0]), int(inlier_pts_source[i][1])), 3, (0, 255, 0), -1)

sum_inlier = sum(N_ini)
sum_second = sum(N_second)
dynamic_image = np.full((image_height,image_width,3),[0,0,255],dtype=np.uint8)

### Find dynamic object
for j in range(cluster_size):
    if N_ini[j] ==0:
        continue
    else :
        module1 = (N_ini[j] - N_first[j])/N_ini[j]*100 > T_reduce
    module2 = N_ini[j]/sum_inlier > T_ini
    module3 = (N_ini[j]/sum_inlier - N_second[j]/sum_second) > 0
    if (module1 and module2) and module3:
        dynamic_image[cluster_idx_image==j] = [0,255,0]

full_image_1_row = np.hstack((images[0],result_image))
full_image_1_row = np.hstack((full_image_1_row,feature_image))
full_image_2_row = np.hstack((inlier_image,inlier_image_second))
full_image_2_row = np.hstack((full_image_2_row,dynamic_image))
full_image = np.vstack((full_image_1_row,full_image_2_row))

cv2.imshow("result image", full_image)
cv2.waitKey(0)