import pylab
import numpy as np
from scipy import linalg
import cv2

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
    
    # Normalized image coordinates
    x1 /= x1[2]
    mean_1 = np.mean(x1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1, x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    
    # Compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)
    
    # Reverse normalization
    F = np.dot(T1.T, np.dot(F, T2))
    
    return F / F[2,2]

def my_print(mat):
    for r in mat:
        print(r)
        
def main():
    x1 = np.random.rand(3,8)
    x2 = np.random.rand(3,8)
    x1[2,:] = 1
    x2[2,:] = 1
    
    print("\n---------Matrix results---------")
    print("Testing x1:")
    my_print(x1.T)
    print("Testing x2:")
    my_print(x2.T)
    
    print("")
          
    A = compute_fundamental(x1, x2)
    print("F by custom function:")
    my_print(A)
    
    A_norm = compute_fundamental_normalized(x1, x2)
    print("F_norm by custom function:")
    my_print(A_norm)
    
    A_cv, mask = cv2.findFundamentalMat(x1.T, x2.T, 2) # 2 --> 8-point algorithm
    print("F by OpenCV package:")
    my_print(A_cv)
    
    print("\n---------Test point results---------")
    
    p = np.random.rand(3,1)
    p[2,:] = 1
    print("Testing a point: ", p[:,0],"\n")

    L = A * p
    print("Epiline by custom function:")
    my_print(L)
    
    L_norm = A_norm * p
    print("Epiline by custom function normalized:")
    my_print(L_norm)
    
    L_cv = A_cv * p
    print("Epiline by OpenCV package")
    my_print(L_cv)
   
if __name__ == "__main__":
    main()