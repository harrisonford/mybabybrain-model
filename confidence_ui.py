from poseModels import HumanPoseModel
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from imageio import imread


def main(subsample=1, path='/home/babybrain/Escritorio/frames_seg_000790/'):
    # we can subsample files in case we want less but fast results, a value of 1 implies no subsample
    full_paths = [path + a_frame for num, a_frame in enumerate(os.listdir(path)) if num % subsample == 0]

    model = HumanPoseModel(input_list=full_paths)
    model.run_model(verbose=True)
    confidences = model.calculate_confidence()

    # for each input/output/confidence trio we'll make a cool image
    # we'll show a heatmap overlay on the original frame on left side plus confidence bars on the right side
    for a_file, an_output, a_confidence in zip(model.input_list, model.outputs, confidences):

        # read image
        image = imread(a_file)

        # calculate part heatmaps
        heatmaps = model.make_heatmaps_once(an_output)

        # create plot canvas
        fig = plt.figure()
        outer_canvas = grid.GridSpec(1, 2)

        # plot heatmaps in first grid
        left_canvas = grid.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_canvas[0])
        # f, axarr = plt.subplots(4, 2)
        for num, a_heatmap in enumerate(heatmaps):
            ax = plt.Subplot(fig, left_canvas[num])
            ax.set_title(model.model_config.all_joints_names[num])
            ax.axis('off')
            ax.imshow(image, interpolation='bilinear')
            ax.imshow(a_heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
            fig.add_subplot(ax)

        # TODO: plot confidence on right canvas and save figs !
        fig.show()
        fig.close()


if __name__ == '__main__':
    main(subsample=480)
