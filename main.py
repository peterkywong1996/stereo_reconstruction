import numpy as np
from scipy import linalg
from math import pow, sqrt
import cv2
import argparse
from matplotlib import pyplot as plt

def load_points(data_dir='./data/points/01_x1.txt'):
    x = np.loadtxt(data_dir, delimiter=',')
    x = np.insert(x, x.shape[1], 1.0, axis=1)
    x = x.T
    return x

def pointsBySIFT(dir1='./data/non-rectified/01/im0.png', dir2='./data/non-rectified/01/im1.png', matcher_choice='bf', filter_points=False, dist_threshold=0.2):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(dir1)
    img2 = cv2.imread(dir2)
    
    (kps1, descs1) = sift.detectAndCompute(img1, None)
    (kps2, descs2) = sift.detectAndCompute(img2, None)
    
    print("Number of points detected by SIFT: ", len(kps1))
    
    matcher = None
    if matcher_choice == 'bf':
        matcher = cv2.BFMatcher()
    elif matcher_choice == 'flann':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
   
    matches = matcher.knnMatch(descs1, descs2, k=2)
    
    good = []
    if filter_points:
        for m,n in matches:
            if m.distance < dist_threshold*n.distance:
                good.append([m])
        matches = good
          
    x1 = np.float32([kps1[m[0].queryIdx].pt for m in matches]).reshape(-1,2)
    x2 = np.float32([kps2[m[0].trainIdx].pt for m in matches]).reshape(-1,2)
    
    x1 = np.insert(x1, x1.shape[1], 1.0, axis=1)
    x2 = np.insert(x2, x2.shape[1], 1.0, axis=1)
    
    x1 = x1.T
    x2 = x2.T
    
    return x1, x2

def compute_transformation(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points do not match.")
        
    # Normalized image coordinates
    x1 /= x1[2]
    x2 /= x2[2]
    mean1 = np.mean(x1[:2], axis=1)
    mean2 = np.mean(x2[:2], axis=1)
    
    dist_sum = 0.0
    for i in range(n):
        dist_sum += pow((x1[0,i]-mean1[0]),2) + pow((x1[1,i]-mean1[1]),2)
        dist_sum += pow((x2[0,i]-mean2[0]),2) + pow((x2[1,i]-mean2[1]),2)
        
    S = 1 / sqrt(dist_sum/(2*2*n))
    
    T1 = np.array([[S,0,-S*mean1[0]],[0,S,-S*mean1[1]],[0,0,1]])
    T2 = np.array([[S,0,-S*mean2[0]],[0,S,-S*mean2[1]],[0,0,1]])
    
    return T1, T2

def compute_fundamental(x1, x2):
    """
    Computes the fundamental matrix from corresponding points 
    (x1,x2 3*n arrays) using the 8 point algorithm.
    Each row in the A matrix below is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points do not match.")
        
    # Build matrix for equations
    A = np.zeros((n,9))
    
    for i in range(n):
        A[i] = [
            x1[0,i]*x2[0,i], x1[1,i]*x2[0,i], x1[2,i]*x2[0,i],
            x1[0,i]*x2[1,i], x1[1,i]*x2[1,i], x1[2,i]*x2[1,i],
            x1[0,i]*x2[2,i], x1[1,i]*x2[2,i], x1[2,i]*x2[2,i]
        ]
        
    # Compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3,3)
    
    # Constrain F: make rank 2 by zeroing out last singullar value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    
    return F / F[2,2]

def compute_fundamental_normalized(x1, x2):
    """
    Computes the fundamental matrix from corresponding points 
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points do not match.")
    
    T1, T2 = compute_transformation(x1, x2)
    
    x1 = np.dot(T1, x1)
    x2 = np.dot(T2, x2)

    # Compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # De-normalization
    F = np.dot(T2.T, np.dot(F, T1))

    return F / F[2,2]

class RansacModel(object):    
    def __init__(self, debug=True, return_all=True):
        self.debug = debug
        self.return_all = return_all
    
    def fit(self,data):        
        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3,:8]
        x2 = data[3:,:8]
        
        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1,x2)
        return F
    
    def get_error(self,data,F):
        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]
        
        # Sampson distance as error measure
        Fx1 = np.dot(F,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = ( np.diag(np.dot(x1.T, np.dot(F, x2))) )**2 / denom 
        
        # return error per point
        return err

def F_from_ransac(x1, x2, model, n, k, t, d, debug=False, return_all=False):
    import ransac
    
    data = np.vstack((x1,x2))
    
    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model, n, k, t, d, debug, return_all)
    
    return F, ransac_data['inliers']

def my_print(mat):
    for r in mat:
        print(r)

def drawlines(img1, img2, lines, pts1, pts2):
    img1_ = img1.copy()
    img2_ = img2.copy()
    r, c = img1_.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1.T,pts2.T):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1_ = cv2.line(img1_, (x0,y0), (x1,y1), color,1)
        img1_ = cv2.circle(img1_,tuple(pt1[:2]),5,color,-1)
        img2_ = cv2.circle(img2_,tuple(pt2[:2]),5,color,-1)
    return img1_,img2_

def drawEpilines(img1,img2,pts1,pts2,F):
    # Calculate and draw the epiplines in img1
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1,3)
    imgLeft,_ = drawlines(img1,img2,lines1,pts1,pts2)

    # Calculate and draw the epiplines in img2
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1,3)
    imgRight,_ = drawlines(img2,img1,lines2,pts2,pts1)

    return imgLeft, imgRight


def main():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--manual', action='store_true', default=False)
    parser.add_argument('--dir_x1', type=str)
    parser.add_argument('--dir_x2', type=str)
    
    parser.add_argument('--dir_img1', type=str, default='./data/non-rectified/01/im0.png')
    parser.add_argument('--dir_img2', type=str, default='./data/non-rectified/01/im1.png')
    parser.add_argument('--point_matcher', type=str, choices=['bf', 'flann'], default='bf')
    parser.add_argument('--filter_points', action='store_true', default=False)
    parser.add_argument('--dist_threshold', type=float, default=0.2)
    parser.add_argument('--use_ransac', action='store_true', default=False)
    parser.add_argument('--min_data', type=int)
    parser.add_argument('--min_close_data', type=int)    
    parser.add_argument('--max_iteration', type=int)
    parser.add_argument('--max_error', type=float)
    args = parser.parse_args()
    
    if args.manual:
        x1 = load_points(args.dir_x1)
        x2 = load_points(args.dir_x2)
        
        print("Testing x1:")
        my_print(x1.T)
        print("Testing x2:")
        my_print(x2.T)
    
    else:
        x1, x2 = pointsBySIFT(args.dir_img1, args.dir_img2, matcher_choice=args.point_matcher, filter_points=args.filter_points, dist_threshold=args.dist_threshold)
        print("Number of points used as good matches: ", x1.shape[1])
       
    if args.use_ransac:
        n = args.min_data
        k = args.max_iteration
        t = args.max_error
        d = args.min_close_data
        
        model = RansacModel(debug=False, return_all=True)
        F_ransac, data_ransac = F_from_ransac(x1, x2, model, n, k, t, d, debug=model.debug, return_all=model.return_all)
    
    N_in = len(data_ransac)
    print('\nNumber of inliers: ', N_in)
    if N_in == x1.shape[1]:
        print("RANSAC failed: did not meet fit acceptance criteria")
    else:
        print("RANSAC succeeded")
    
    x1_inliers = x1[:, data_ransac]
    x2_inliers = x2[:, data_ransac]
    
    x1_best8 = x1[:, data_ransac[:8]]
    x2_best8 = x2[:, data_ransac[:8]]
    
    print("\n---------------Matrix results---------------")
    
    print("")
    print("\n---1. By OpenCV function---")
    
    print("\n(1a) F_cv with best 8 points:")
    F_cv_best8, mask = cv2.findFundamentalMat(x1_best8.T, x2_best8.T, 2) # 2 --> 8-point algorithm
    my_print(F_cv_best8)
    
    print("\n(1b) F_cv with all {} inliers:".format(N_in))
    F_cv_sift, mask = cv2.findFundamentalMat(x1_inliers.T, x2_inliers.T, 2) # 2 --> 8-point algorithm
    my_print(F_cv_sift)
    
    print("")
    print("\n---2. With normalization---")
    
    print("\n(2a) F_norm with best 8 points:")
    F_norm_best8 = compute_fundamental_normalized(x1_best8, x2_best8)
    my_print(F_norm_best8)
    
    print("\n(2b) F_norm with all {} inliers:".format(N_in))
    F_norm_sift = compute_fundamental_normalized(x1_inliers, x2_inliers)
    my_print(F_norm_sift)
    
    print("")
    print("\n---3. Without normalization---")
    
    print("\n(3a) F with best 8 points:")
    F_unnorm_best8 = compute_fundamental(x1_best8, x2_best8)
    my_print(F_unnorm_best8)
    
    print("\n(3b) F with all {} inliers:".format(N_in))
    F_unnorm_sift = compute_fundamental(x1_inliers, x2_inliers)
    my_print(F_unnorm_sift)
    
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    
    img1 = cv2.imread(args.dir_img1)
    img2 = cv2.imread(args.dir_img2)

    imgCvBest8Left, imgCvBest8Right = drawEpilines(img1, img2, x1.T, x2.T, F_cv_best8)
    imgCvSiftLeft, imgCvSiftRight = drawEpilines(img1, img2, x1_inliers.T, x2_inliers.T, F_cv_sift)

    imgNormBest8Left, imgNormBest8Right = drawEpilines(img1, img2, x1.T, x2.T, F_norm_best8)
    imgNormSiftLeft, imgNormSiftRight = drawEpilines(img1, img2, x1_inliers.T, x2_inliers.T, F_norm_sift)

    imgUnnormBest8Left, imgUnnormBest8Right = drawEpilines(img1, img2, x1.T, x2.T, F_unnorm_best8)
    imgUnnormSiftLeft, imgUnnormSiftRight = drawEpilines(img1, img2, x1_inliers.T, x2_inliers.T, F_unnorm_sift)

    plt.subplot(2,2,1),plt.imshow(imgCvSiftLeft)
    plt.subplot(2,2,2),plt.imshow(imgCvSiftRight)

    plt.subplot(2,2,3),plt.imshow(imgNormSiftLeft)
    plt.subplot(2,2,4),plt.imshow(imgNormSiftRight)
    plt.show()

    print("imgCvSiftLeft/imgNormSiftLeft? {}".format(
        np.count_nonzero(imgCvSiftLeft != imgNormSiftLeft) > 0
    ))
    
if __name__ == "__main__":
    main()