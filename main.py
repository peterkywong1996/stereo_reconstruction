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

def pointsBySIFT(dir1='./data/non-rectified/01/im0.png', dir2='./data/non-rectified/01/im1.png', toFilter=True, matcher_choice='bf'):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(dir1)
    img2 = cv2.imread(dir2)
    
    (kps1, descs1) = sift.detectAndCompute(img1, None)
    (kps2, descs2) = sift.detectAndCompute(img2, None)
    
    print("Number of points detected: ", len(kps1))
    
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
    if toFilter:
        for m,n in matches:
            if m.distance < 0.2*n.distance:
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
    r, c = img1.shape[:2]
    for r,pt1,pt2 in zip(lines,pts1.T,pts2.T):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1[:2]),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2[:2]),5,color,-1)
    return img1,img2

def main():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--manual', action='store_true', default=False)
    parser.add_argument('--dir_x1', type=str)
    parser.add_argument('--dir_x2', type=str)
    
    parser.add_argument('--dir_img1', type=str, default='./data/non-rectified/01/im0.png')
    parser.add_argument('--dir_img2', type=str, default='./data/non-rectified/01/im1.png')
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
        x1, x2 = pointsBySIFT(args.dir_img1, args.dir_img2)
        print("Number of points used: ", x1.shape[1])
    
    print("\n---------Matrix results---------")
    
    F = compute_fundamental(x1, x2)
    print("\nF by custom function:")
    my_print(F)
    
    F_norm = compute_fundamental_normalized(x1, x2)
    print("\nF_norm by custom function:")
    my_print(F_norm)
    
    F_cv, mask = cv2.findFundamentalMat(x1.T, x2.T, 2) # 2 --> 8-point algorithm
    print("\nF by OpenCV package:")
    my_print(F_cv)
    
    print("")
    
    if args.use_ransac:
        n = args.min_data
        k = args.max_iteration
        t = args.max_error
        d = args.min_close_data
        
        model = RansacModel(debug=False, return_all=True)
        F_ransac, data_ransac = F_from_ransac(x1, x2, model, n, k, t, d, debug=model.debug, return_all=model.return_all)
                
        print("\nF by RANSAC trained with 8 points:")
        my_print(F_ransac)
        
        print('\nNumber of inliners: ', len(data_ransac))
        x1 = x1[:, data_ransac]
        x2 = x2[:, data_ransac]
        
        F_ransac = compute_fundamental_normalized(x1, x2)
        print("\nF by RANSAC with all best inliners:")
        my_print(F_ransac)
    
    """
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    
    img1 = cv2.imread(args.dir_img1)
    img2 = cv2.imread(args.dir_img2)
    
    lines1 = cv2.computeCorrespondEpilines(x2.T, 2, F_norm)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,x1,x2)
    
    lines2 = cv2.computeCorrespondEpilines(x1.T, 1, F_norm)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,x2,x1)
    
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    """
    
if __name__ == "__main__":
    main()