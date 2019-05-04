import argparse
import cv2
import numpy as np


def add_padding(input, padding):
    rows = input.shape[0]
    columns = input.shape[1]
    channels = input.shape[2]
    
    output = np.zeros((rows + padding * 2, columns + padding * 2, channels), dtype=float)
    output[ padding : rows + padding, padding : columns + padding, :] = input
    return output


def search_bounds(column, block_size, width, rshift):
    disparity_range = 75
    padding = block_size // 2
    right_bound = column
    if rshift:
        left_bound = column - disparity_range
        if left_bound < padding:
            left_bound = padding
        step = 1
    else:
        left_bound = column + disparity_range
        if left_bound >= (width - 2*padding):
            left_bound = width - 2*padding - 2
        step = -1
    return left_bound, right_bound, step


# max disparity 30
def disparity_map(left, right, block_size, rshift):
    """
    padding = block_size // 2
    left_img = add_padding(left, padding)
    right_img = add_padding(right, padding)
    """
    
    left_img = left
    right_img = right
    height, width, channels = left_img.shape

    # d_map = np.zeros((height - padding*2, width - padding*2), dtype=float)
    d_map = np.zeros(left.shape , dtype=float)

    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):

            bestdist = float('inf')
            shift = 0
            left_pixel = left_img[row:row + block_size, col:col + block_size]
            l_bound, r_bound, step = search_bounds(col, block_size, width, rshift)

            # for i in range(l_bound, r_bound - padding*2):
            for i in range(l_bound, r_bound, step):
                right_pixel = right_img[row:row + block_size, i:i + block_size]

                # if euclid_dist(left_pixel, right_pixel) < bestdist :
                ssd = np.sum((left_pixel - right_pixel) ** 2)
                # print('row:',row,' col:',col,' i:',i,' bestdist:',bestdist,' shift:',shift,' ssd:',ssd)
                if ssd < bestdist:
                    bestdist = ssd
                    shift = i

            if rshift:
                d_map[row, col] = col - shift
            else:
                d_map[row, col] = shift - col
            print('Calculated Disparity at ('+str(row)+','+str(col)+') :', d_map[row,col])

    return d_map


def main():   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir_img1', type=str, default='./data/rectified/01/im0.png')
    parser.add_argument('--dir_img2', type=str, default='./data/rectified/01/im1.png')   
    parser.add_argument('--block_size', type=int, default=8)
        
    args = parser.parse_args()
    
    left = cv2.imread(args.dir_img1)
    right = cv2.imread(args.dir_img2)
    
    """
    padding = block_size // 2
    left_img = add_padding(left, padding)
    right_img = add_padding(right, padding)
    """
    
    left_img = left
    right_img = right
    height, width, channels = left_img.shape
    
    block_size = args.block_size
    rshift = False
    d_map = np.zeros(left.shape , dtype=float)
    
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):
            bestdist = float('inf')
            shift = 0
            left_pixel = left_img[row:row + block_size, col:col + block_size]
            l_bound, r_bound, step = search_bounds(col, block_size, width, rshift)
            
            # for i in range(l_bound, r_bound - padding*2):
            for i in range(l_bound, r_bound, step):
                right_pixel = right_img[row:row + block_size, i:i + block_size]

                # if euclid_dist(left_pixel, right_pixel) < bestdist :
                ssd = np.sum((left_pixel - right_pixel) ** 2)
                # print('row:',row,' col:',col,' i:',i,' bestdist:',bestdist,' shift:',shift,' ssd:',ssd)
                if ssd < bestdist:
                    bestdist = ssd
                    shift = i

            if rshift:
                d_map[row, col] = col - shift
            else:
                d_map[row, col] = shift - col
            print('Calculated Disparity at ('+str(row)+','+str(col)+') :', d_map[row,col])
    
    cv2.imshow('disparity', d_map)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()