import json
import numpy as np
from poseModels import HumanPoseModel, PoseEstimationModel
from HumanPose.nnet.predict import argmax_pose_predict
import matplotlib.pyplot as plt
from imageio import imread
import tensorflow as tf


# import json annotations as a dictionary
# TODO: JSON processing should be done in mybabybrain-database!
# TODO: Create a function that renames part names (faster this way than marking)
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

    # get annotations for each frame using a part as reference
    annotations = load_annotations(annotation_path)
    frame_list, _, _ = get_xy_for('l-elbow', annotations)

    # use human model to get joint names to use
    model_human = HumanPoseModel()
    joint_list = model_human.model_config.all_joints_names

    # get x, y annotations
    _, x_anno_l, y_anno_l = get_xy_lr(annotations, joint_list, ignore_with='r-')
    _, x_anno_r, y_anno_r = get_xy_lr(annotations, joint_list, ignore_with='l-')

    # run session for each frame image annotated for both models
    x_human = np.empty([len(frame_list), len(joint_list)])
    y_human = np.empty([len(frame_list), len(joint_list)])
    x_pose = np.empty([len(frame_list), len(joint_list)])
    y_pose = np.empty([len(frame_list), len(joint_list)])
    image_dimensions = []

    # human pose
    for index, a_frame in enumerate(frame_list):

        scmap, locref = model_human.run_model_once(frames_path + a_frame)
        image = imread(frames_path + a_frame, as_gray=True)
        image_dimensions.append(image.shape)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = argmax_pose_predict(scmap, locref, model_human.model_config.stride)

        for joint_index, joint_name in enumerate(joint_list):

            x_human[index, joint_index] = pose[joint_index, 0]
            y_human[index, joint_index] = pose[joint_index, 1]

    # pose-est
    tf.reset_default_graph()
    model_pose = PoseEstimationModel()
    for index, a_frame in enumerate(frame_list):

        an_output = model_pose.run_model_once(frames_path + a_frame)
        # TODO: resolve what to do when pose-est detects many humans
        if len(an_output) > 1:  # then store only the first human
            an_output = an_output[0]

        body_parts = an_output.body_parts
        for a_part in body_parts:
            if a_part in model_pose.model_config.all_joints_list.keys():
                part_true_index = model_pose.model_config.all_joints_list[a_part]
                x_norm = body_parts[a_part].x
                y_norm = body_parts[a_part].y
                # TODO check xy order in dimensions
                x = x_norm * image_dimensions[index][0]
                y = y_norm * image_dimensions[index][1]
                x_pose[index, part_true_index] = y
                y_pose[index, part_true_index] = x

    # now calculate distances
    distances_r_human = calculate_distances(x_human, y_human, x_anno_r, y_anno_r, image_dim=image_dimensions)
    distances_l_human = calculate_distances(x_human, y_human, x_anno_l, y_anno_l, image_dim=image_dimensions)
    distances_r_pose = calculate_distances(x_pose, y_pose, x_anno_r, y_anno_r, image_dim=image_dimensions)
    distances_l_pose = calculate_distances(x_pose, y_pose, x_anno_l, y_anno_l, image_dim=image_dimensions)

    # merge the best distance results
    # TODO: pose and human results should have same dimensions right?
    distances_human = np.empty(distances_l_human.shape)
    distances_pose = np.empty(distances_l_pose.shape)
    for i in range(distances_human.shape[0]):
        for j in range(distances_human.shape[1]):
            distances_human[i, j] = min(distances_l_human[i, j], distances_r_human[i, j])
            distances_pose[i, j] = min(distances_l_pose[i, j], distances_r_pose[i, j])

    distance_steps_human, rates_human = detection_rate(distances_human, nsteps=50)
    distance_steps_pose, rates_pose = detection_rate(distances_pose, nsteps=50)
    rates_human = rates_human*100
    rates_pose = rates_pose*100

    # plot all joints graph
    # human
    fig, ax = plt.subplots()
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection %')
    ax.set_title('Performance HumanPose: Threshold vs Detection Rate')
    ax.set_xlim([0, 1])
    for joint_index, joint_name in enumerate(joint_list):
        ax.plot(distance_steps_human[joint_index], rates_human[joint_index], label=joint_name)
    ax.legend(loc='upper right')
    plt.savefig('/home/babybrain/Escritorio/performances_all_bodyparts_human.png')
    plt.close()

    # pose-est
    fig, ax = plt.subplots()
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection %')
    ax.set_title('Performance PoseEst: Threshold vs Detection Rate')
    ax.set_xlim([0, 1])
    for joint_index, joint_name in enumerate(joint_list):
        ax.plot(distance_steps_pose[joint_index], rates_pose[joint_index], label=joint_name)
    ax.legend(loc='upper right')
    plt.savefig('/home/babybrain/Escritorio/performances_all_bodyparts_pose.png')
    plt.close()

    # average performance of joints
    average_distances_human = np.nanmean(distance_steps_human, axis=0)
    average_distances_pose = np.nanmean(distance_steps_pose, axis=0)
    average_ratio_human = np.nanmean(rates_human, axis=0)
    average_ration_pose = np.nanmean(rates_pose, axis=0)

    # finally plot the graph
    fig, ax = plt.subplots()
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Detection %')
    ax.set_title('Average Distance threshold vs Average Detection Ratio')
    ax.set_xlim([0, 1])

    ax.plot(average_distances_human, average_ratio_human, label='HumanPose')
    ax.plot(average_distances_pose, average_ration_pose, label='PoseEst')
    ax.legend()
    plt.savefig('/home/babybrain/Escritorio/performances_bodyparts.png')
    plt.show()


if __name__ == '__main__':
    main()
