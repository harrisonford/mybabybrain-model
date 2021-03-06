import poseModels
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import warnings
import pandas as pd
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


def save_frame_result(output_file, image_matrix, heatmaps, confidences, part_names, model_name, errors=None):

    # create plot canvas
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("{} Tracking and Confidence levels in a sample video".format(model_name))
    outer_canvas = grid.GridSpec(1, 2)

    # plot heatmaps in first grid
    left_canvas = grid.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_canvas[0])
    for num in range(len(confidences)):
        if heatmaps is None:
            a_heatmap = None
        else:
            a_heatmap = heatmaps[num]
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
    ax.bar(index, np.abs(confidences), bar_width, alpha=opacity, color='b', yerr=errors)  # TODO: PoseEst more conf
    ax.set_ylim((0, 1))
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Confidence Values for Each Joint')
    ax.set_xticks(index)
    ax.set_xticklabels(part_names)
    fig.add_subplot(ax)
    if output_file is not None:
        plt.savefig(output_file, format='png')
    plt.close()
    return fig


def main(input_video, output_video, nstop=999999, model_name='HumanPose'):

    # reset the graph
    tf.reset_default_graph()
    # create the human pose model
    if model_name == 'PoseEst':
        model = poseModels.PoseEstimationModel()
    else:
        model = poseModels.HumanPoseModel()
    # container to save confidences
    confidences = []

    # extract each frame
    vidcap = cv2.VideoCapture(input_video)
    success, image = vidcap.read()
    # for each input/output/confidence trio we'll make a cool image
    # we'll show a heatmap overlay on the original frame on left side plus confidence bars on the right side
    # video_dimensions = (image.shape[1], image.shape[0])
    video_dimensions = (1200, 800)
    fps = 240.0
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), fps, video_dimensions)
    while success:
        # apply model to frame
        an_output = model._run_model_once(image)
        model.outputs.append(an_output)
        # calculate heat-map
        a_heatmap_set = model.make_heatmaps_once(an_output)
        # calculate confidence set
        a_confidence_set = model.calculate_confidence_once(an_output)
        confidences.append(a_confidence_set)

        this_frame = vidcap.get(0)
        if this_frame > nstop:
            break
        print("saving frame {}".format(this_frame))
        # transform fig into rgb
        fig = save_frame_result(None, image, a_heatmap_set, a_confidence_set, model.joint_names, model_name)
        canvas = FigureCanvas(fig)
        canvas.draw()
        canvas_str = canvas.tostring_rgb()
        frame = np.fromstring(canvas_str, dtype=np.uint8, sep='')
        # TODO: video is not saving because it has to be opencv image format
        frame = frame.reshape(canvas.get_width_height()[::-1] + (3, ))
        # write it in BGR format
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # go agane
        success, image = vidcap.read()

    # clean up
    print("cleaning up, saving...")
    cv2.destroyAllWindows()
    video.release()
    # save confidences
    df = pd.DataFrame(confidences, columns=['1', '2', '3', '4', '5', '6', '7', '8'])
    df.to_csv('./confidences.csv')
    # make final image
    final_image_output = './final_image.png'
    confidence_matrix = np.array(confidences)
    save_frame_result(final_image_output, image, None, np.nanmean(confidence_matrix, axis=0), model.joint_names,
                      model_name, errors=np.nanstd(confidence_matrix, axis=0))


if __name__ == '__main__':
    # parse arguments input and output video
    # input_path = str(sys.argv[1])
    # output_path = str(sys.argv[2])
    input_path = '/home/harrisonford/Videos/babybrain/000345.MP4'
    output_path = './sample.avi'
    main(input_path, output_path, nstop=240*1, model_name='PoseEst')
