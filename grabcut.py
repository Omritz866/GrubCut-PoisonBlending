import numpy as np
import cv2
import argparse

# My imports
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import igraph as ig
from matplotlib import pyplot as plt

GC_BGD = 0  # Hard background pixel
GC_FGD = 1  # Hard foreground pixel (not used)
GC_PR_BGD = 2  # Soft background pixel
GC_PR_FGD = 3  # Soft foreground pixel

CONVERGENCE_THRESHOLD = 700  # Convergence default threshold

# Define the 8-neighborhood offset for each pixel
pixel_neighbor_offsets = [(-1, 0),  (1, 0),   (0, -1),  (0, 1),   (-1, -1),  (-1, 1),  (1, -1),  (1, 1)  ]


flattened_image = np.float32()  # The image flattened to a 2D
img_height, img_width = -1, -1  #height and width of the image
total_pixel_count = -1  # no. of pixels
beta_value = -1
pixel_neighbors_dict = {}  # Dictionary storing pixel neighbors
total_neighbors_count = 0  # no. of neighbors

n_link_capacity_list = []  # N-link capacities
neighbor_sum_list = []
previous_energy = -1


""" Get an image,Calculate the neighbors of each pixel and return a Dictionary where the keys are the pixels and values are arrays of their neighbors. """
def calculate_pixel_neighbors(image):
    y_indices, x_indices = np.meshgrid(np.arange(img_height), np.arange(img_width), indexing='ij')
    pixel_indices = y_indices * img_width + x_indices

    # Flatten the arrays for easier manipulation
    y_indices_flat = y_indices.ravel()
    x_indices_flat = x_indices.ravel()
    pixel_indices_flat = pixel_indices.ravel()

    # Initialize the dictionary for neighbors
    pixel_neighbors = {}

    for pixel in range(len(pixel_indices_flat)):
        neighbors = []
        y, x = y_indices_flat[pixel], x_indices_flat[pixel]

        for dy, dx in pixel_neighbor_offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < img_height and 0 <= nx < img_width:
                neighbor_index = ny * img_width + nx
                neighbors.append(neighbor_index)

        pixel_neighbors[pixel_indices_flat[pixel]] = neighbors

    return pixel_neighbors



""" Get an image and Calculate N-links capacities """
def calculate_n_link_capacities(image):
    global n_link_capacity_list
    global neighbor_sum_list
    global pixel_neighbors_dict

    neighbor_sum_list = np.zeros(flattened_image.shape[0])

    pixel_neighbors_dict = calculate_pixel_neighbors(image)

    for pixel, neighbors in enumerate(pixel_neighbors_dict.values()):
        delta_values = (flattened_image[pixel] - flattened_image[neighbors])
        N_values = (50 / (np.linalg.norm(pixel - np.array(neighbors).reshape(1, len(neighbors)), axis=0)) *
                    np.exp(-beta_value * np.diag(delta_values.dot(delta_values.T))))
        n_link_capacity_list.extend(N_values)
        neighbor_sum_list[pixel] += np.sum(N_values)
"""   Get an image ,Calculate and return the beta """
def calculate_beta(image):
    global total_neighbors_count
    global beta_value

    # Initialize total distance
    total_distance = 0

    # Initialize neighbors count
    neighbor_count_matrix = np.zeros((img_height, img_width), dtype=int)

    for dy, dx in pixel_neighbor_offsets:
        shifted_image = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)

        valid_mask = (
                (np.arange(img_height)[:, None] + dy >= 0) & (np.arange(img_height)[:, None] + dy < img_height) &
                (np.arange(img_width) + dx >= 0) & (np.arange(img_width) + dx < img_width)
        )

        dist = np.sum((image - shifted_image) ** 2, axis=2)
        valid_diff = np.where(valid_mask, dist, 0)

        total_distance += np.sum(valid_diff)
        neighbor_count_matrix += valid_mask.astype(int)

    total_neighbors_count = np.sum(neighbor_count_matrix)
    beta_value = 1 / (2 * total_distance / total_neighbors_count)
"""  get an image , Calculate and return the D(n) for a GMM  """
def calculate_dn(gmm_model: GaussianMixture, colors):
    sigma = 0

    for component in range(gmm_model.n_components):
        weight = gmm_model.weights_[component]
        mean_value = gmm_model.means_[component]
        covariance_matrix = gmm_model.covariances_[component]

        det_cov = np.linalg.det(covariance_matrix)
        inv_cov = np.linalg.inv(covariance_matrix)
        dist_from_mean = colors - mean_value

        inner_exp = np.einsum('ij,ij->i', dist_from_mean, np.dot(inv_cov, dist_from_mean.T).T)

        sigma += (weight / np.sqrt(det_cov)) * np.exp(-0.5 * inner_exp)

    return -1 * np.log(sigma)

""" Get an image,pixels and gmm_model and Update a GMM based on pixels """
def update_gmm_model(image, pixels, gmm_model: GaussianMixture):
    num_components = gmm_model.n_components
    num_features = image.shape[-1]

    kmeans = KMeans(n_clusters=num_components, n_init='auto')
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    covariances = np.zeros((num_components, num_features, num_features))
    weights = np.zeros(num_components)

    for component in range(num_components):
        weights[component] = np.sum(labels == component) / len(pixels)
        covariances[component] = 0.01 * np.eye(num_features)

        indices = np.where(labels == component)[0]
        if indices.size > 0:
            X = pixels[indices]
            N = len(indices)
            mean_value = centroids[component]
            covariance_matrix = (1 / N) * np.dot((X - mean_value).T, (X - mean_value))
            covariances[component] = covariance_matrix

    gmm_model.means_ = centroids
    gmm_model.covariances_ = covariances
    gmm_model.weights_ = weights

    return gmm_model

""" Get a mask, BG and FG gaussian muxture and Calculate the T-link capacities of the background and the foreground """
def calculate_t_link_capacities(mask, bg_gmm: GaussianMixture, fg_gmm: GaussianMixture, regularization=1e-6):
    mask_flattened = mask.reshape(-1)

    # Calculate the largest weight in the graph
    max_weight = max(neighbor_sum_list)

    source_capacities = np.where(mask_flattened == GC_FGD, max_weight, 0)
    target_capacities = np.where(mask_flattened == GC_BGD, max_weight, 0)

    bg_dn = calculate_dn(bg_gmm, flattened_image[(mask_flattened != GC_BGD) & (mask_flattened != GC_FGD)])
    fg_dn = calculate_dn(fg_gmm, flattened_image[(mask_flattened != GC_BGD) & (mask_flattened != GC_FGD)])

    source_capacities[(mask_flattened != GC_BGD) & (mask_flattened != GC_FGD)] = bg_dn
    target_capacities[(mask_flattened != GC_BGD) & (mask_flattened != GC_FGD)] = fg_dn

    return source_capacities.tolist(), target_capacities.tolist()


"""Run the GrabCut algorithm """
def grabcut_algorithm(image, rect, num_iterations=10):
    # Convert image to float ndarray
    image = np.asarray(image, dtype=np.float64)

    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute coordinates
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD  # The center of the rectangle

    # Initialize GMMs
    bg_gmm, fg_gmm = initialize_gmms(image, mask)

    # Initialize previous energy to infinity (so we won't converge after one iteration)
    previous_energy = float('inf')

    for i in range(num_iterations):
        # Update GMM
        bg_gmm, fg_gmm = update_gmms(image, mask, bg_gmm, fg_gmm)

        # Calculate mincut and energy
        mincut_sets, energy = calculate_mincut(image, mask, bg_gmm, fg_gmm)

        # Update mask
        mask = update_mask(mincut_sets, mask)

        # Check for convergence, end the process if converged
        if check_convergence(energy, previous_energy):
            break

        # Update current energy to be previous energy for the next iteration
        previous_energy = energy

    # Return the final mask and the GMMs
    return mask, bg_gmm, fg_gmm
""" Get an image,mask,BG and FG GMM and Update the GMMs """
def update_gmms(image, mask, bg_gmm, fg_gmm):
    # Separate background and foreground pixels (based on the current mask)
    bg_pixels = image[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = image[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)]

    # Update each GMM individually
    update_gmm_model(image, bg_pixels, bg_gmm)
    update_gmm_model(image, fg_pixels, fg_gmm)

    return bg_gmm, fg_gmm
"""Get an image,mask,number of components for the GMMs ,Initialize and return two GMM models"""

def initialize_gmms(image, mask, num_components=5):
    global flattened_image
    global img_height
    global img_width
    global total_pixel_count

    flattened_image = np.float32(image).reshape(-1, 3)

    # Calculate the size of the picture
    img_height, img_width = image.shape[:2]
    total_pixel_count = img_height * img_width

    # Initialize GMMs using K-means clusters
    bg_gmm = GaussianMixture(n_components=num_components, covariance_type='full')  # Background GMM
    fg_gmm = GaussianMixture(n_components=num_components, covariance_type='full')  # Foreground GMM

    # Calculate beta
    calculate_beta(image)

    # Initialize N-link capacities
    calculate_n_link_capacities(image)

    return bg_gmm, fg_gmm



"""Get an image , mask , BG and FG GMMs ,Calculate the min-cut of the graph based on the current mask and GMMs"""

def calculate_mincut(image, mask, bg_gmm, fg_gmm):
    # Calculate the source and target capacities for the graph
    source_capacities, target_capacities = calculate_t_link_capacities(mask, bg_gmm, fg_gmm)

    # Combine all capacities
    capacities = source_capacities + target_capacities + n_link_capacity_list

    # Create a graph where every pixel is a vertex with 2 additional vertices for source and target
    graph = ig.Graph(total_pixel_count + 2)

    # Set source and target indices
    source, target = total_pixel_count, total_pixel_count + 1

    # Calculate edges
    neighbor_edges = [(pixel, neighbor) for pixel in pixel_neighbors_dict for neighbor in pixel_neighbors_dict[pixel]]  # Pixel's edges
    source_edges = np.column_stack((np.full(total_pixel_count, source), np.arange(total_pixel_count)))  # Source edges
    target_edges = np.column_stack((np.arange(total_pixel_count), np.full(total_pixel_count, target)))  # Target edges

    # Add the edges to the graph, where every pixel is connected to the source, the target and its 8 neighbors
    graph.add_edges(np.concatenate((source_edges, target_edges)).tolist() + neighbor_edges)

    # Calculate the minimum cut of the graph
    min_cut = graph.st_mincut(source, target, capacities)

    # Extract the energy (the value of the minimum cut)
    energy = min_cut.value

    return [min_cut.partition[0], min_cut.partition[1]], energy


""" Calculate Jaccard and accuracy  similarity for the predicted mask and ground truth mask"""
def calculate_metrics(predicted_mask, gt_mask):
    # Flatten the masks
    predicted_mask_flat = predicted_mask.flatten()
    gt_mask_flat = gt_mask.flatten()

    # Calculate number of correct pixels
    correct_pixels = np.sum(predicted_mask_flat == gt_mask_flat)
    accuracy = correct_pixels / (np.size(predicted_mask_flat))

    # Calculate intersection and union of predicted and gt masks
    intersection = np.sum(predicted_mask_flat & gt_mask_flat == 1)
    union = np.sum(predicted_mask_flat | gt_mask_flat)

    # Calculate Jaccard similarity
    if union == 0:  # If both masks are 0 (prevent crashes)
        jaccard_similarity = 0
    else:
        jaccard_similarity = intersection / union

    return accuracy, jaccard_similarity

""" Update the mask based on the mincut results """
def update_mask(mincut_sets, mask: np.ndarray):
    mask_height, mask_width = mask.shape

    pr_pixels = np.where((mask == GC_PR_BGD) | (mask == GC_PR_FGD))
    img_pixels = np.arange(mask_height * mask_width, dtype=np.uint32).reshape(mask_height, mask_width)

    # Calculate new mask where every pixel that isn't a hard pixel get reassigned a value, based on the given mincut
    mask[pr_pixels] = np.where(np.isin(img_pixels[pr_pixels], mincut_sets[0]), GC_PR_FGD, GC_PR_BGD)

    return mask


def check_convergence(energy, previous_energy, threshold=CONVERGENCE_THRESHOLD):
    # Calculate the change in energy
    energy_change = np.abs(energy - previous_energy)

    return energy_change < threshold


""" Parse the user-given command line arguments"""
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana2', help='name of the image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='path to your own image file')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='change the rect (x,y,w,h) if needed')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse_arguments()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bg_gmm, fg_gmm = grabcut_algorithm(img, rect)
    mask[mask == GC_PR_BGD] = GC_BGD  # All pixels classified as BG are given HARD BG value.
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = calculate_metrics(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image
    img_cut = img * (mask[:, :, np.newaxis])

    # Save the images to disk instead of using cv2.imshow
    cv2.imwrite('original_image.jpg', img)
    cv2.imwrite('grabcut_mask.jpg', 255 * mask)
    cv2.imwrite('grabcut_result.jpg', img_cut)

    # Use matplotlib to display the images
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    img_original = cv2.imread('original_image.jpg')
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    # GrabCut Mask
    plt.subplot(1, 3, 2)
    img_mask = cv2.imread('grabcut_mask.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_mask, cmap='gray')
    plt.title('GrabCut Mask')

    # GrabCut Result
    plt.subplot(1, 3, 3)
    img_result = cv2.imread('grabcut_result.jpg')
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.title('GrabCut Result')

    # Display all images
    plt.show()

    # Cleanup saved images if necessary
    import os
    os.remove('original_image.jpg')
    os.remove('grabcut_mask.jpg')
    os.remove('grabcut_result.jpg')
    os.remove('grabcut_result.jpg')
