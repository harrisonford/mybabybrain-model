import poseModels
import os
from imageio import imread
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import cv2
import warnings
plt.rcParams.update({'font.size': 10})


def save_frame_result_dual(output_file, image_matrix, dual_heatmaps, dual_confidences, part_names, model_names):
    # create plot canvas
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Tracking and Confidence levels in a sample video ({} | {})".format(model_names[0], model_names[1]))
    outer_canvas = grid.GridSpec(1, 2)

    # plot heatmaps in first grid
    left_canvas = grid.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_canvas[0])
    for num, heatmap1 in enumerate(dual_heatmaps[0]):
        heatmap2 = dual_heatmaps[1][num]
        confidences1 = dual_confidences[0][num]
        confidences2 = dual_confidences[1][num]
        ax = plt.Subplot(fig, left_canvas[num])
        ax.set_title(part_names[num] + " = {0:.2f} | {1:.2f}".format(confidences1, confidences2))
        ax.axis('off')
        ax.imshow(image_matrix, interpolation='bilinear')
        if np.any(heatmap1) and np.any(heatmap2):
            ax.imshow(heatmap1, alpha=0.5, cmap='jet', interpolation='bilinear')
        else:  # warn about no joint detection
            warnings.warn("No heatmap found for joint {} in frame {}".format(part_names[num], output_file))

        fig.add_subplot(ax)

    # plot confidences in second grid (dual)
    right_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[1])
    ax = plt.Subplot(fig, right_canvas[0])
    bar_width = 0.35
    opacity = 0.4
    n_groups = len(dual_confidences[0])
    index = np.arange(n_groups)
    ax.bar(index - bar_width/2, dual_confidences[0], bar_width, alpha=opacity, color='b', label=model_names[0])
    ax.bar(index + bar_width/2, dual_confidences[1], bar_width, alpha=opacity, color='r', label=model_names[1])
    ax.set_ylim((0, 1))
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Confidence Values for Each Joint')
    ax.set_xticks(index)
    ax.set_xticklabels(part_names)
    ax.legend()
    fig.add_subplot(ax)
    plt.savefig(output_file, format='png')
    plt.close()


def save_frame_result(output_file, image_matrix, heatmaps, confidences, part_names, model_name):

    # create plot canvas
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("{} Tracking and Confidence levels in a sample video".format(model_name))
    outer_canvas = grid.GridSpec(1, 2)

    # plot heatmaps in first grid
    left_canvas = grid.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_canvas[0])
    for num, a_heatmap in enumerate(heatmaps):
        ax = plt.Subplot(fig, left_canvas[num])
        ax.set_title(part_names[num] + " = {0:.2f}".format(confidences[num]))
        ax.axis('off')
        ax.imshow(image_matrix, interpolation='bilinear')
        if np.any(a_heatmap):
            ax.imshow(a_heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
        else:  # warn about no joint detection
            warnings.warn("No heatmap found for joint {} in frame {} with model {}".format(
                part_names[num], output_file, model_name
            ))

        fig.add_subplot(ax)

    # plot confidences in second grid
    right_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[1])
    ax = plt.Subplot(fig, right_canvas[0])
    bar_width = 0.35
    opacity = 0.4
    n_groups = len(confidences)
    index = np.arange(n_groups)
    ax.bar(index, confidences, bar_width, alpha=opacity, color='b')
    ax.set_ylim((0, 1))
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Confidence Values for Each Joint')
    ax.set_xticks(index)
    ax.set_xticklabels(part_names)
    fig.add_subplot(ax)
    plt.savefig(output_file, format='png')
    plt.close()


def main(subsample=1, n_stop=24000,
         path='/home/babybrain/Escritorio/frames_seg_000790/',
         outpath='/home/babybrain/Escritorio/frames_seg_000790_results/'):
    # we can subsample files in case we want less but fast results, a value of 1 implies no subsample
    file_names = sorted(os.listdir(path))
    file_names = [a_frame for num, a_frame in enumerate(file_names) if num % subsample == 0 and num <= n_stop]
    full_paths = [path + a_frame for a_frame in file_names]

    # run models
    model_human = poseModels.HumanPoseModel(input_list=full_paths)
    model_human.run_model(verbose=True)
    confidences_human = model_human.calculate_confidence()

    tf.reset_default_graph()
    model_pose = poseModels.PoseEstimationModel(input_list=full_paths)
    model_pose.run_model(verbose=True)
    confidences_pose = model_pose.calculate_confidence()

    # for each input/output/confidence trio we'll make a cool image
    # we'll show a heatmap overlay on the original frame on left side plus confidence bars on the right side
    for index, a_file in enumerate(file_names):

        # read image
        image = imread(full_paths[index])

        # calculate part heatmaps
        heatmaps_human = model_human.make_heatmaps_once(model_human.outputs[index])
        heatmaps_pose = model_pose.make_heatmaps_once(model_pose.outputs[index])

        # confidence for output
        conf_human = confidences_human[index]
        conf_pose = confidences_pose[index]

        # save result frames
        outpath_human = outpath + 'HumanPose/' + a_file
        outpath_pose = outpath + 'PoseEst/' + a_file
        outpath_dual = outpath + 'Both/' + a_file

        print("saving frame {}".format(a_file))
        save_frame_result(outpath_human, image, heatmaps_human, conf_human,
                          model_human.joint_names, model_human.model_name)
        save_frame_result(outpath_pose, image, heatmaps_pose, conf_pose,
                          model_human.joint_names, model_pose.model_name)
        save_frame_result_dual(outpath_dual, image, [heatmaps_human, heatmaps_pose], [conf_human, conf_pose],
                               model_human.joint_names, [model_human.model_name, model_pose.model_name])

    # plot final image
    fig = plt.figure(figsize=(12, 8))
    outer_canvas = grid.GridSpec(1, 2)

    # plot a frame on the left
    used_frame = model_human.input_list[0]
    image = imread(used_frame)
    left_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[0])
    ax = plt.Subplot(fig, left_canvas[0])
    ax.axis('off')
    ax.imshow(image, interpolation='bilinear')
    fig.add_subplot(ax)

    # plot average confidences to the right
    right_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[1])
    ax = plt.Subplot(fig, right_canvas[0])

    # human
    confidences = np.array(confidences_human)
    confidences_mean = np.nanmean(confidences, axis=0)
    confidences_std = np.nanstd(confidences, axis=0)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    n_groups = len(confidences_mean)
    index = np.arange(n_groups)
    ax.bar(index, confidences_mean, bar_width, alpha=opacity, color='b', yerr=confidences_std,
           error_kw=error_config, label='Human-Pose')
    # pose-est
    confidences = np.array(confidences_pose)
    confidences_mean = np.nanmean(confidences, axis=0)
    confidences_std = np.nanstd(confidences, axis=0)
    ax.bar(index + bar_width, confidences_mean, bar_width, alpha=opacity, color='r', yerr=confidences_std,
           error_kw=error_config, label='Pose-Est')

    ax.set_ylim(0, 1)
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Mean Confidence Values in one video')
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels(model_human.model_config.all_joints_names)
    ax.legend()

    fig.add_subplot(ax)
    plt.savefig("{}{}.png".format(outpath, "0"), format='png')
    plt.close()

    # finally create a sample video from all frames for each model
    average_frames = 300
    # human
    frame_list = [outpath + 'HumanPose/' + a_file for a_file in file_names]
    average_frame = cv2.imread(outpath + '0.png')
    im_dim = (average_frame.shape[1], average_frame.shape[0])
    video = cv2.VideoWriter('{}sample_video_human.avi'.format(outpath),
                            cv2.VideoWriter_fourcc(*'MJPG'), 60, im_dim)
    for a_frame in frame_list:
        frame = cv2.imread(a_frame)
        video.write(frame)
    # also write the average_frame many times
    for i in range(average_frames):
        video.write(average_frame)
    # release writer
    cv2.destroyAllWindows()
    video.release()

    # pose
    frame_list = [outpath + 'PoseEst/' + a_file for a_file in file_names]
    im_dim = (average_frame.shape[1], average_frame.shape[0])
    video = cv2.VideoWriter('{}sample_video_pose.avi'.format(outpath),
                            cv2.VideoWriter_fourcc(*'MJPG'), 60, im_dim)
    for a_frame in frame_list:
        frame = cv2.imread(a_frame)
        video.write(frame)
    # also write the average_frame many times
    for i in range(average_frames):
        video.write(average_frame)
    # release writer
    cv2.destroyAllWindows()
    video.release()

    # both
    frame_list = [outpath + 'Both/' + a_file for a_file in file_names]
    im_dim = (average_frame.shape[1], average_frame.shape[0])
    video = cv2.VideoWriter('{}sample_video_both.avi'.format(outpath),
                            cv2.VideoWriter_fourcc(*'MJPG'), 60, im_dim)
    for a_frame in frame_list:
        frame = cv2.imread(a_frame)
        video.write(frame)
    # also write the average_frame many times
    for i in range(average_frames):
        video.write(average_frame)
    # release writer
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main(subsample=1, n_stop=1)
