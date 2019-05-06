import os
import argparse
import numpy as np
import cv2


def computeDisparityBySSD(left_img, right_img, block_size, numDisparities):
    d_map = np.zeros(left_img.shape, dtype=float)
    height, width, channels = left_img.shape
    
    for i in range(height):
        for j in range(block_size + numDisparities, width - block_size + 1):
            ssd = np.empty([numDisparities, 1])
            l = left_img[i, (j - block_size):(j + block_size)]
            
            for d in range(numDisparities):
                r = right_img[i, (j - d - block_size):(j - d + block_size)]
                ssd[d] = np.sum((l[:,:]-r[:,:])**2)
            
            d_map[i, j] = np.argmin(ssd)
            
        print("Completed {} row".format(str(i)))
    
    d_map = cv2.normalize(src=d_map, dst=d_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    d_map = np.uint8(d_map)
    
    return d_map


def computeDisparityBySGBM(left_img, right_img, filter_params, args):
    sgbm_left = cv2.StereoSGBM_create(
        minDisparity=args.minDisparity,
        numDisparities=args.numDisparities,
        blockSize=args.block_size,
        uniquenessRatio=args.uniquenessRatio,
        speckleWindowSize=args.speckleWindowSize,
        speckleRange=args.speckleRange,
        disp12MaxDiff=args.disp12MaxDiff,
        P1=8*3*args.block_size**2,
        P2=32*3*args.block_size**2
    )
    sgbm_right = cv2.ximgproc.createRightMatcher(sgbm_left)
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm_left)
    wls_filter.setLambda(filter_params['lambda'])
    wls_filter.setSigmaColor(filter_params['sigma'])
        
    d_map_left = sgbm_left.compute(left_img, right_img)
    d_map_right = sgbm_right.compute(left_img, right_img)
    
    d_map_left = np.int16(d_map_left)
    d_map_right = np.int16(d_map_right)
    
    d_map = wls_filter.filter(d_map_left, left_img, None, d_map_right)
    
    d_map = cv2.normalize(src=d_map, dst=d_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    d_map = np.uint8(d_map)

    return d_map


def show_disparity(d_map):
    cv2.imshow('disparity', d_map)
    cv2.waitKey(0)
    

def save_disparity(result_dir, fname, d_map):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    save_dir = os.path.join(result_dir, fname)
    np.save(save_dir, d_map)
    print('Disparity map saved as ', save_dir)
    
    
def main():   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir_img1', type=str, default='./data/rectified/01/im0.png')
    parser.add_argument('--dir_img2', type=str, default='./data/rectified/01/im1.png')
    parser.add_argument('--result_dir', type=str, default='./result/01')
    parser.add_argument('--use_ssd', action='store_true', default=False)
    # if using SSD, only block_size has effect
    
    parser.add_argument('--block_size', type=int, default=5)
    parser.add_argument('--minDisparity', type=int, default=0)
    parser.add_argument('--numDisparities', type=int, default=16)
    parser.add_argument('--disp12MaxDiff', type=int, default=1)
    parser.add_argument('--uniquenessRatio', type=int, default=10)
    parser.add_argument('--speckleWindowSize', type=int, default=100)
    parser.add_argument('--speckleRange', type=int, default=32)
        
    args = parser.parse_args()
    
    left_img = cv2.imread(args.dir_img1)
    right_img = cv2.imread(args.dir_img2)
    
    if args.use_ssd:
        d_map_ssd = computeDisparityBySSD(left_img, right_img, args.block_size, args.numDisparities)
        save_disparity(args.result_dir, 'ssd', d_map_ssd)
        show_disparity(d_map_ssd)
    
    # FILTER Parameters
    filter_params = {
        'lambda': 80000,
        'sigma': 1.2
    }
    
    d_map_sgbm = computeDisparityBySGBM(left_img, right_img, filter_params, args)
    
    save_disparity(args.result_dir, 'sgbm', d_map_sgbm)
    show_disparity(d_map_sgbm)

if __name__ == "__main__":
    main()