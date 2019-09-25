import poseModels
import os
import sys
from imageio import imread
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import cv2
import warnings
import shutil
plt.rcParams.update({'font.size': 10})


# pads both matrices to be the same size, assuming they're both centered and 2-d
def extend_matrices(matrix1, matrix2):

    # calculate length, width of m1, m2
    dims_m1 = matrix1.shape
    dims_m2 = matrix2.shape
    h1, w1 = dims_m1[1]/2, dims_m1[0]/2
    h2, w2 = dims_m2[1]/2, dims_m2[0]/2

    # calculate horizontal/vertical padding for matrices
    h_pad1 = max(w2 - w1, 0)
    v_pad1 = max(h2 - h1, 0)

    h_pad2 = max(w1 - w2, 0)
    v_pad2 = max(h1 - h2, 0)

    # do the padding
    padded_matrix1 = np.pad(matrix1, ((int(np.ceil(h_pad1)), int(np.trunc(h_pad1))),
                                      (int(np.ceil(v_pad1)), int(np.trunc(v_pad1)))), 'constant')
    padded_matrix2 = np.pad(matrix2, ((int(np.ceil(h_pad2)), int(np.trunc(h_pad2))),
                                      (int(np.ceil(v_pad2)), int(np.trunc(v_pad2)))), 'constant')
    return padded_matrix1, padded_matrix2


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
            ax.imshow(heatmap2, alpha=0.5, cmap='jet', interpolation='bilinear')
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
    # TODO: np.absolute to manage L-R for pose-estimation model, should do better
    ax.bar(index - bar_width/2, np.absolute(dual_confidences[0]), bar_width, alpha=opacity,
           color='b', label=model_names[0])
    ax.bar(index + bar_width/2, np.absolute(dual_confidences[1]), bar_width, alpha=opacity,
           color='r', label=model_names[1])
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
    ax.bar(index, np.absolute(confidences), bar_width, alpha=opacity, color='b')  # TODO: np.abs because L-R on PoseEst
    ax.set_ylim((0, 1))
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Confidence Values for Each Joint')
    ax.set_xticks(index)
    ax.set_xticklabels(part_names)
    fig.add_subplot(ax)
    plt.savefig(output_file, format='png')
    plt.close()


def main(input_video, output_video, nstop=999999):

    # create the human pose model
    model = poseModels.HumanPoseModel()

    # extract each frame
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    # for each input/output/confidence trio we'll make a cool image, as output we'll use a _temp folder
    # we'll show a heatmap overlay on the original frame on left side plus confidence bars on the right side
    temp_path = './_temp'
    os.mkdir(temp_path)
    video_dimensions = (image.shape[1], image.shape[0])
    fps = 240
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, video_dimensions)
    while success:
        # apply model to frame
        an_output = model._run_model_once(image)
        model.outputs.append(an_output)
        # calculate heat-map
        a_heatmap_set = model.make_heatmaps_once(an_output)
        # calculate confidence set
        a_confidence_set = model.calculate_confidence_once(an_output)

        this_frame = vidcap.get(0)
        if this_frame > nstop:
            break
        print("saving frame {}".format(this_frame))
        temp_out = "{}/{}.png".format(temp_path, this_frame)
        save_frame_result(temp_out, image, a_heatmap_set, a_confidence_set, model.joint_names, 'HumanPose')
        success, image = vidcap.read()

        frame = cv2.imread(temp_out)
        video.write(frame)

    # clean up
    cv2.destroyAllWindows()
    video.release()
    shutil.rmtree(temp_path)


if __name__ == '__main__':
    # parse arguments input and output video
    # input_path = str(sys.argv[1])
    # output_path = str(sys.argv[2])
    input_path = '/home/harrisonford/Videos/babybrain/000345.MP4'
    output_path = './sample.avi'
    main(input_path, output_path, nstop=100)
