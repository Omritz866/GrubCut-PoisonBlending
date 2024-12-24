import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
from matplotlib import pyplot as plt


def poisson_blend(source_img, target_img, mask_img, blend_center):
    # Get the dimensions of the source image
    src_height, src_width, _ = source_img.shape

    # Get the offset for placing the source image in the target image
    # Calculate the Laplacian of the source image
    y_offset, x_offset = blend_center[1] - src_height // 2, blend_center[0] - src_width // 2
    laplacian_src = cv2.Laplacian(source_img, cv2.CV_64F)

    # Get indices of the pixels inside the mask
    mask_indices = np.where(mask_img != 0)
    num_mask_pixels = len(mask_indices[0])

    # Create sparse matrix for the Poisson equation
    sparse_A = scipy.sparse.lil_matrix((num_mask_pixels, num_mask_pixels))
    b_channel_r = np.zeros(num_mask_pixels)
    b_channel_g = np.zeros(num_mask_pixels)
    b_channel_b = np.zeros(num_mask_pixels)

    # Create a mapping from pixel indices to matrix rows
    index_mapping = np.zeros(mask_img.shape, dtype=np.int32)
    index_mapping[mask_indices] = np.arange(num_mask_pixels)

    for i, (y_coord, x_coord) in enumerate(zip(mask_indices[0], mask_indices[1])):
        sparse_A[i, i] = 4
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_y, neighbor_x = y_coord + dy, x_coord + dx
            if 0 <= neighbor_y < src_height and 0 <= neighbor_x < src_width:
                if mask_img[neighbor_y, neighbor_x] != 0:
                    sparse_A[i, index_mapping[neighbor_y, neighbor_x]] = -1
                else:
                    b_channel_r[i] += target_img[neighbor_y + y_offset, neighbor_x + x_offset, 0]
                    b_channel_g[i] += target_img[neighbor_y + y_offset, neighbor_x + x_offset, 1]
                    b_channel_b[i] += target_img[neighbor_y + y_offset, neighbor_x + x_offset, 2]

        b_channel_r[i] -= laplacian_src[y_coord, x_coord, 0]
        b_channel_g[i] -= laplacian_src[y_coord, x_coord, 1]
        b_channel_b[i] -= laplacian_src[y_coord, x_coord, 2]

    # Solve the Poisson equation
    sparse_A = sparse_A.tocsr()
    result_r = spsolve(sparse_A, b_channel_r)
    result_g = spsolve(sparse_A, b_channel_g)
    result_b = spsolve(sparse_A, b_channel_b)

    # Blend the result into the target image
    blended_img = target_img.copy()
    for i, (y_coord, x_coord) in enumerate(zip(mask_indices[0], mask_indices[1])):
        blended_img[y_coord + y_offset, x_coord + x_offset, 0] = np.clip(result_r[i], 0, 255)
        blended_img[y_coord + y_offset, x_coord + x_offset, 1] = np.clip(result_g[i], 0, 255)
        blended_img[y_coord + y_offset, x_coord + x_offset, 2] = np.clip(result_b[i], 0, 255)

    return blended_img


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='target image file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    target_image = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    source_image = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        mask_image = np.full(source_image.shape, 255, dtype=np.uint8)
    else:
        mask_image = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)[1]

    blend_center = (int(target_image.shape[1] / 2), int(target_image.shape[0] / 2))

    cloned_image = poisson_blend(source_image, target_image, mask_image, blend_center)

    cv2.imwrite('cloned_output.jpg', cloned_image)

# plot the image
display_img = cv2.imread('cloned_output.jpg')
plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
plt.title('Blended Image')
plt.axis('off')  # Turn off axis labels
plt.show()


import os
os.remove('cloned_output.jpg')
