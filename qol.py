# A collection of quality of life functions that may be used in many places
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


# A silly function to set a default value if not in kwargs
def kwarget(key, default, **kwargs):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


# Get all frames contained in frame vector from a video file path
# TODO: Assuming notations are sorted from min to max
def get_frames_from(video_path, frame_vector, threshold=0, framerate=240, verbose=False):
    # create an open-cv video capture object
    image_container = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        getvalue = vidcap.get(0)  # value in milliseconds
        if verbose:
            print("loading time {} from {}".format(getvalue/1000, video_path))
        if almost_in(getvalue*framerate/1000, frame_vector, threshold=threshold):
            image_container.append(image)
            frame_vector.pop(0)  # assuming value found is first value
        success, image = vidcap.read()
    return image_container


# Returns true if value is in list with a threshold value of drift
def almost_in(some_value, a_list, threshold=0):
    inside = False
    for list_value in a_list:
        inside = inside or (np.abs(list_value - some_value) <= threshold)
    return inside


# Transform times to vector indexes
def times_to_frames(time_vector, framerate=240):
    frame_vector = [a_time*framerate for a_time in time_vector]
    return frame_vector


def load_annotations(json_path):

    with open(json_path) as f:
        annotations = json.load(f)
    return annotations


# load x,y for all joints in list, use 'r-' or 'l-' to filter left or right respectively
def get_xy_lr(annotations, joint_list, ignore_with='l-'):

    frame_elements = []
    # TODO: fix this messy way of discerning data !
    frame_candidates = annotations['_via_img_metadata']  # may or may not have data
    for key, elements in frame_candidates.items():
        if elements['regions']:  # then it has annotations
            frame_elements.append(elements)

    # now we know how many frames have data
    x = np.empty([len(frame_elements), len(joint_list)])
    y = np.empty([len(frame_elements), len(joint_list)])
    frames = []

    for frame_index, element in enumerate(frame_elements):
        region_list = element['regions']
        frames.append(element['filename'])
        for joint_index, joint_name in enumerate(joint_list):
            for region in region_list:
                region_name = region['region_attributes']['id']
                if joint_name in region_name and ignore_with not in region_name:
                    x[frame_index, joint_index] = region['shape_attributes']['cx']
                    y[frame_index, joint_index] = region['shape_attributes']['cy']

    return np.array(frames), np.array(x), np.array(y)


# load x, y for a certain data id (r-ankle, l-shoulder, etc)
def get_xy_for(part, annotations):

    frame = []
    x = []
    y = []
    frames = annotations['_via_img_metadata']
    for key, elements in frames.items():
        if elements['regions']:  # then it has annotations
            frame.append(elements['filename'])
            region_list = elements['regions']
            for region in region_list:
                if region['region_attributes']['id'] == part:
                    x.append(region['shape_attributes']['cx'])
                    y.append(region['shape_attributes']['cy'])

    return np.array(frame), np.array(x), np.array(y)


# calculate distances of output array of limb compared to ground truth
# both arrays should be same length
def calculate_distances(x_array, y_array, truth_x_array, truth_y_array, normalized=True, image_dim=None):

    distances = [np.hypot(abs(x1-x2), abs(y1-y2)) for x1, y1, x2, y2 in zip(x_array, y_array,
                                                                            truth_x_array, truth_y_array)]
    distances = np.array(distances)

    if normalized and image_dim is not None:
        # TODO: resolve how to normalize detection distance
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = distances[i, j]/np.hypot(image_dim[i][0]/2, image_dim[i][1]/2)
        # distances = (distances - distances.min())/(distances.max() - distances.min())
    return distances


# from distances array calculate the detection rate vs (normalized) distance data to plot
def detection_rate(distance_matrix, nsteps=10, normalized=True):

    steps_matrix = []
    rates_matrix = []
    for joint_index in range(distance_matrix.shape[1]):
        joint_distance = distance_matrix[:, joint_index]
        distance_steps = np.linspace(0, joint_distance.max(), nsteps)
        rates = np.empty(len(distance_steps))

        for index, a_distance in enumerate(distance_steps):
            rates[index] = np.sum(joint_distance < a_distance)
        rates = np.array(rates)

        if normalized:
            rates = rates / len(joint_distance)
        steps_matrix.append(distance_steps)
        rates_matrix.append(rates)
    return np.array(steps_matrix), np.array(rates_matrix)


# given the ground-truth and detection compute the delta difference error
def compute_error(dt_path, gt_path, normalize=None, threshold=0.250):
    dt = load_annotations(dt_path)
    gt = load_annotations(gt_path)

    # get keypoints as np array from dt and gt and calculate distances
    distances = []
    visible = []
    for gt_sample in gt['annotations']:
        # find the dt_sample associated to the gt_sample
        for a_dt in dt:
            if np.abs(a_dt['image_id'] - gt_sample['id']) <= threshold:
                dt_sample = a_dt
                dt_keypoints = np.array(dt_sample['keypoints'])
                gt_keypoints = np.array(gt_sample['keypoints'])

                # separate by x, y and visible value
                dt_x = dt_keypoints[0::3]
                dt_y = dt_keypoints[1::3]

                gt_x = gt_keypoints[0::3]
                gt_y = gt_keypoints[1::3]
                gt_visible = gt_keypoints[2::3]

                # calculate deltas and apply normalize factor
                norm_factor = 1
                if normalize == 'body':
                    # get bbox
                    bbox = gt_sample['bbox']
                    # norm factor is diagonal of box (w, h)
                    norm_factor = 1/np.hypot(bbox[2], bbox[3])
                delta_x = np.abs(dt_x - gt_x)
                delta_y = np.abs(dt_y - gt_y)

                distances.append(np.hypot(delta_x, delta_y)*norm_factor)
                visible.append(gt_visible)
    return distances, visible


# compare machine learning model to baseline performance with a ROC
def evaluate_model(predictions, probs, train_predictions, train_probs, train_labels, test_labels, output=None):
    baseline = {}
    baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    results = {}
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    for metric in ['recall', 'precision', 'roc']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)}'
            f' Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')

    if output:
        plt.savefig(output)
