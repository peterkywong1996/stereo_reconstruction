import os
import argparse
import numpy as np
import cv2


def get_img_shape(img):
    if len(img.shape) == 2:
        return img.shape[0], img.shape[1], None
    elif len(img.shape) == 3:
        return img.shape
    else:
        return [None]*3
    
def add_padding(img, padding):   
    height, width, channels = get_img_shape(img)
    if channels:
        output = np.zeros((height, width + padding, channels), dtype=float)
    else:
        output = np.zeros((height, width + padding), dtype=float)
    
    output[:, padding:] = img
    
    return output.astype(np.uint8)


def computeDisparityBySSD(img_L, img_R, block_size, numDisparities):   
    height, width, _ = get_img_shape(img_L)
    disparity_map = np.zeros(img_L.shape, dtype=float)
        
    for h in range(height):
        for w in range(block_size + numDisparities, width - block_size + 1):
            ssd = np.empty([numDisparities, 1])
            block_L = img_L[h, (w - block_size):(w + block_size)]
            
            for d in range(numDisparities):
                block_R = img_R[h, (w - d - block_size):(w - d + block_size)]
                ssd[d] = np.sum(( block_L[:,:] - block_R[:,:] ) ** 2)
            
            disparity_map[h, w] = np.argmin(ssd)
    
    disparity_map = cv2.normalize(src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)
    
    return disparity_map


def computeDisparityBySGBM(img_L, img_R, filter_params, args):   
    sgbm_L = cv2.StereoSGBM_create(
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
    sgbm_R = cv2.ximgproc.createRightMatcher(sgbm_L)
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=sgbm_L)
    wls_filter.setLambda(filter_params['lambda'])
    wls_filter.setSigmaColor(filter_params['sigma'])
        
    disparity_map_L = sgbm_L.compute(img_L, img_R)
    disparity_map_R = sgbm_R.compute(img_L, img_R)
    
    disparity_map_L = np.int16(disparity_map_L)
    disparity_map_R = np.int16(disparity_map_R)
    
    disparity_map = wls_filter.filter(disparity_map_L, img_L, None, disparity_map_R)
    
    disparity_map = cv2.normalize(src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity_map = np.uint8(disparity_map)

    return disparity_map


def show_disparity(disparity_map):
    cv2.imshow('disparity', disparity_map)
    cv2.waitKey(0)
    

def save_disparity(result_dir, fname, disparity_map):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    save_dir = os.path.join(result_dir, fname)
    np.save(save_dir, disparity_map)
    cv2.imwrite(save_dir+'.png', disparity_map)
    
    print('Disparity map saved as ', save_dir)
    
    
def main():   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir_img1', type=str, default='./data/rectified/01/im0.png')
    parser.add_argument('--dir_img2', type=str, default='./data/rectified/01/im1.png')
    parser.add_argument('--result_dir', type=str, default='./result/01')
    parser.add_argument('--use_ssd', action='store_true', default=False)
    # SSD is influenced by block_size and numDisparities only
    
    parser.add_argument('--block_size', type=int, default=5)
    parser.add_argument('--minDisparity', type=int, default=0)
    parser.add_argument('--numDisparities', type=int, default=16)
    parser.add_argument('--disp12MaxDiff', type=int, default=1)
    parser.add_argument('--uniquenessRatio', type=int, default=10)
    parser.add_argument('--speckleWindowSize', type=int, default=100)
    parser.add_argument('--speckleRange', type=int, default=32)
        
    args = parser.parse_args()
    
    img_L = cv2.imread(args.dir_img1)
    img_R = cv2.imread(args.dir_img2)
    
    padding = args.numDisparities
    img_L = add_padding(img_L, padding)
    img_R = add_padding(img_R, padding)

    if args.use_ssd:
        disparity_map_ssd = computeDisparityBySSD(img_L, img_R, args.block_size, args.numDisparities)
        disparity_map_ssd = disparity_map_ssd[:, padding:]
        save_disparity(args.result_dir, 'ssd', disparity_map_ssd)
        show_disparity(disparity_map_ssd)
    
    # FILTER Parameters
    filter_params = {
        'lambda': 80000,
        'sigma': 1.2
    }
    
    disparity_map_sgbm = computeDisparityBySGBM(img_L, img_R, filter_params, args)
    disparity_map_sgbm = disparity_map_sgbm[:, padding:]
    save_disparity(args.result_dir, 'sgbm', disparity_map_sgbm)
    show_disparity(disparity_map_sgbm)

if __name__ == "__main__":
    main()