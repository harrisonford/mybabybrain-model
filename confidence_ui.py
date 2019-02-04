import poseModels
import os
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
plt.rcParams.update({'font.size': 10})


def main(subsample=1, n_stop=24000,
         path='/home/babybrain/Escritorio/frames_seg_000790/',
         outpath='/home/babybrain/Escritorio/frames_seg_000790_results/'):
    # we can subsample files in case we want less but fast results, a value of 1 implies no subsample
    file_names = os.listdir(path)
    full_paths = [path + a_frame for num, a_frame in enumerate(file_names) if num % subsample == 0 and num <= n_stop]

    model = poseModels.PoseEstimationModel(input_list=full_paths)
    model.run_model(verbose=True)
    confidences = model.calculate_confidence()

    # for each input/output/confidence trio we'll make a cool image
    # we'll show a heatmap overlay on the original frame on left side plus confidence bars on the right side
    for an_input, an_output, a_confidence, a_file in zip(model.input_list, model.outputs, confidences, file_names):

        # read image
        image = imread(an_input)

        # calculate part heatmaps
        heatmaps = model.make_heatmaps_once(an_output)

        # create plot canvas
        fig = plt.figure(figsize=(12, 8))
        outer_canvas = grid.GridSpec(1, 2)

        # plot heatmaps in first grid
        left_canvas = grid.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_canvas[0])
        for num, a_heatmap in enumerate(heatmaps):
            ax = plt.Subplot(fig, left_canvas[num])
            ax.set_title(model.model_config.all_joints_names[num])
            ax.axis('off')
            ax.imshow(image, interpolation='bilinear')
            ax.imshow(a_heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
            fig.add_subplot(ax)

        # plot confidences in second grid
        right_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[1])
        ax = plt.Subplot(fig, right_canvas[0])
        bar_width = 0.35
        opacity = 0.4
        n_groups = len(a_confidence)
        index = np.arange(n_groups)
        ax.bar(index, a_confidence, bar_width, alpha=opacity, color='b', label='Human-Pose-Segmented')
        ax.set_ylim((0, 2))
        ax.set_xlabel('Body-Part')
        ax.set_ylabel('Confidence Value')
        ax.set_title('Confidence Values in one video')
        ax.set_xticks(index)
        ax.set_xticklabels(model.model_config.all_joints_names)
        fig.add_subplot(ax)
        plt.savefig(outpath + a_file, format='png')
        plt.close()

    # plot final image
    fig = plt.figure(figsize=(12, 8))
    outer_canvas = grid.GridSpec(1, 2)

    # plot a frame on the left
    used_frame = model.input_list[0]
    image = imread(used_frame)
    left_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[0])
    ax = plt.Subplot(fig, left_canvas[0])
    ax.axis('off')
    ax.imshow(image, interpolation='bilinear')
    fig.add_subplot(ax)

    # plot average confidences to the right
    right_canvas = grid.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_canvas[1])
    ax = plt.Subplot(fig, right_canvas[0])
    confidences = np.array(confidences)
    confidences_mean = np.nanmean(confidences, axis=0)
    confidences_std = np.nanstd(confidences, axis=0)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    n_groups = len(confidences_mean)
    index = np.arange(n_groups)

    ax.bar(index, confidences_mean, bar_width, alpha=opacity, color='b', yerr=confidences_std,
           error_kw=error_config, label='Human-Pose')

    ax.set_ylim(0, 2)
    ax.set_xlabel('Body-Part')
    ax.set_ylabel('Confidence Value')
    ax.set_title('Mean Confidence Values in one video')
    ax.set_xticks(index)
    ax.set_xticklabels(model.model_config.all_joints_names)
    ax.legend()

    fig.add_subplot(ax)
    plt.savefig("{}{}".format(outpath, "0"), format='png')
    plt.close()


if __name__ == '__main__':
    main(subsample=1, n_stop=60)
