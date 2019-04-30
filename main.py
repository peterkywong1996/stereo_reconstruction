import numpy as np
from scipy import linalg
from math import pow, sqrt
import cv2
import argparse

def load_points(data_dir='./data/points/01_x1.txt'):
    x = np.loadtxt(data_dir, delimiter=',')
    x = x.T
    n = x.shape[1]
    x = np.append(x, np.ones(n))
    x = x.reshape(3,n)
    return x

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

def my_print(mat):
    for r in mat:
        print(r)
        
def main():
    """
    N = 200
    x1 = N*np.random.rand(3,8)
    x2 = N*np.random.rand(3,8)
    x1[2,:] = 1
    x2[2,:] = 1
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_x1', type=str)
    parser.add_argument('--dir_x2', type=str)
    args = parser.parse_args()
    
    x1 = load_points(args.dir_x1)
    x2 = load_points(args.dir_x2)
    
    print("\n---------Matrix results---------")
    print("Testing x1:")
    my_print(x1.T)
    print("Testing x2:")
    my_print(x2.T)
    
    print("")
          
    F = compute_fundamental(x1, x2)
    print("F by custom function:")
    my_print(F)
    
    F_norm = compute_fundamental_normalized(x1, x2)
    print("F_norm by custom function:")
    my_print(F_norm)
    
    F_cv, mask = cv2.findFundamentalMat(x1.T, x2.T, 2) # 2 --> 8-point algorithm
    print("F by OpenCV package:")
    my_print(F_cv)
    
    """
    print("\n---------Test point results---------")
    
    p = N*np.random.rand(3,1)
    p[2,:] = 1
    print("Testing a point: ", p[:,0],"\n")

    L = F * p
    print("Epiline by custom function:")
    my_print(L)
    
    L_norm = F_norm * p
    print("Epiline by custom function normalized:")
    my_print(L_norm)
    
    L_cv = F_cv * p
    print("Epiline by OpenCV package")
    my_print(L_cv)
    """
   
if __name__ == "__main__":
    main()