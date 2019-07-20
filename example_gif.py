import os
from imageio import imread, imsave
from poseModels import PoseEstimationModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


if __name__ == '__main__':

    # use a segmentation of babies, coords are [x0, y0, w, h]
    fixed_width = 350
    fixed_height = 400
    good_baby = dict(coords=[450, 750, fixed_width, fixed_height], name='00345')
    bad_baby = dict(coords=[400, 750, fixed_width, fixed_height], name='00845')

    # segment frames
    good_baby_path = '/home/babybrain/Descargas/000345/'
    bad_baby_path = '/home/babybrain/Descargas/000845/'

    good_baby_outpath = '/home/babybrain/Descargas/000345_segmented/'
    bad_baby_outpath = '/home/babybrain/Descargas/000845_segmented/'

    good_baby_framelist = sorted(os.listdir(good_baby_path))
    bad_baby_framelist = sorted(os.listdir(bad_baby_path))

    for good_baby_frame, bad_baby_frame in zip(good_baby_framelist, bad_baby_framelist):
        good_frame = imread(good_baby_path + good_baby_frame)
        bad_frame = imread(bad_baby_path + bad_baby_frame)

        # subframe
        good_coords = good_baby['coords']
        bad_coords = bad_baby['coords']

        sub_good_frame = good_frame[good_coords[0]:good_coords[0] + good_coords[3],
                                    good_coords[1]:good_coords[1] + good_coords[2], :]

        sub_bad_frame = bad_frame[bad_coords[0]:bad_coords[0] + bad_coords[3],
                                  bad_coords[1]:bad_coords[1] + bad_coords[2], :]

        # save frames
        imsave(good_baby_outpath + good_baby_frame, sub_good_frame)
        imsave(bad_baby_outpath + bad_baby_frame, sub_bad_frame)

    # run the tracking model
    good_baby_inputs = [good_baby_outpath + a_dir for a_dir in good_baby_framelist]
    good_baby_model = PoseEstimationModel(input_list=good_baby_inputs)
    bad_baby_inputs = [bad_baby_outpath + a_dir for a_dir in bad_baby_framelist]
    bad_baby_model = PoseEstimationModel(input_list=bad_baby_inputs)

    good_baby_model.run_model()
    tf.reset_default_graph()
    bad_baby_model.run_model()

    # create the frame + heatmap image
    good_baby_heatpath = '/home/babybrain/Descargas/000345_heatmaps/'
    bad_baby_heatpath = '/home/babybrain/Descargas/000845_heatmaps/'

    for index, (good_baby_frame, bad_baby_frame) in enumerate(zip(good_baby_framelist, bad_baby_framelist)):

        print("Creating heatmap frame {}".format(index))

        # load frame images
        good_frame = imread(good_baby_outpath + good_baby_frame)
        bad_frame = imread(bad_baby_outpath + bad_baby_frame)

        # load, average and resize heatmaps (2 times smaller)
        good_output = good_baby_model.outputs[index]
        bad_output = bad_baby_model.outputs[index]

        # create plotlib image and superpose heatmap
        # good baby
        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(good_frame, interpolation='bilinear')
        if good_output is not None:
            sample_heatmap = good_output.heatmaps[:, :, 0]
            good_heatmap = imresize(np.zeros(sample_heatmap.shape), 2.0, interp='bicubic')
            # good_heatmap = np.zeros(sample_heatmap.shape)
            n_heatmaps = good_output.heatmaps.shape
            for num in range(n_heatmaps[2]):
                # good_heatmap += good_output.heatmaps[:, :, num]
                good_heatmap += imresize(good_output.heatmaps[:, :, num], 2.0, interp='bicubic')
            # good_heatmap = imresize(good_heatmap, 2.0, interp='bicubic')
            ax.imshow(good_heatmap, alpha=0.5, cmap='jet', interpolation='bilinear')
        plt.savefig(good_baby_heatpath + good_baby_frame)
        plt.close()

        # bad baby
        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(bad_frame, interpolation='bilinear')
        if bad_output is not None:
            bad_heatmap = imresize(np.zeros(bad_output.heatmaps[:, :, 0].shape), 2.0, interp='bicubic')
            n_heatmaps = bad_output.heatmaps.shape
            for num in range(n_heatmaps[2]):
                bad_heatmap += imresize(bad_output.heatmaps[:, :, num], 2.0, interp='bicubic')
            ax.imshow(bad_heatmap, alpha=0.5, cmap='jet', interpolation='bicubic')
        plt.savefig(bad_baby_heatpath + bad_baby_frame)
        plt.close()
