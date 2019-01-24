import json
import numpy as np
from poseModels import HumanPoseModel
from HumanPose.nnet.predict import argmax_pose_predict
import matplotlib.pyplot as plt
from imageio import imread


# import json annotations as a dictionary
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


def main():

    # paths to setup
    annotation_path = '/home/babybrain/Escritorio/300145_via.json'
    frames_path = '/home/babybrain/Escritorio/300145/'

    # get annotations for each frame
    annotations = load_annotations(annotation_path)
    frame_list, _, _ = get_xy_for('r-elbow', annotations)

    # get the x,y model prediction for each frame annotated
    model = HumanPoseModel()
    joint_list = model.model_config.all_joints_names

    # get x, y annotations
    _, x_anno_l, y_anno_l = get_xy_lr(annotations, joint_list, ignore_with='r-')
    _, x_anno_r, y_anno_r = get_xy_lr(annotations, joint_list, ignore_with='l-')

    # run session for each frame image annotated
    x_model = np.empty([len(frame_list), len(joint_list)])
    y_model = np.empty([len(frame_list), len(joint_list)])
    image_dimensions = []

    for index, a_frame in enumerate(frame_list):

        scmap, locref = model.run_model_once(frames_path + a_frame)
        image = imread(frames_path + a_frame, as_gray=True)
        image_dimensions.append(image.shape)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = argmax_pose_predict(scmap, locref, model.model_config.stride)

        for joint_index, joint_name in enumerate(joint_list):

            x_model[index, joint_index] = pose[joint_index, 0]
            y_model[index, joint_index] = pose[joint_index, 1]

    # now calculate distances
    distances_r = calculate_distances(x_model, y_model, x_anno_r, y_anno_r, image_dim=image_dimensions)
    distances_l = calculate_distances(x_model, y_model, x_anno_l, y_anno_l, image_dim=image_dimensions)

    # merge the best distance results
    distances = np.empty(distances_l.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i, j] = min(distances_l[i, j], distances_r[i, j])

    distance_steps, rates = detection_rate(distances, nsteps=50)
    rates = rates*100

    # finally plot the graph
    fig, ax = plt.subplots()
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection %')
    ax.set_title('Distance threshold vs Detection Ratio')
    ax.set_xlim([0, 0.5])

    for joint_index, joint_name in enumerate(joint_list):
        ax.plot(distance_steps[joint_index], rates[joint_index], label=joint_name)

    ax.legend()
    plt.savefig('/home/babybrain/Escritorio/performances_bodyparts.png')
    plt.show()


if __name__ == '__main__':
    main()
