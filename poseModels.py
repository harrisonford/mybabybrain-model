from qol import kwarget
import yaml
from easydict import EasyDict
from HumanPose.nnet.predict import setup_pose_prediction, extract_cnn_output
from HumanPose.dataset.pose_dataset import data_to_input
import HumanPose.default_config as default
from PoseEstimation.tf_pose import common
from PoseEstimation.tf_pose.networks import model_wh, get_graph_path
from PoseEstimation.tf_pose.estimator import TfPoseEstimator
from scipy.misc import imread, imresize
import numpy as np
import argparse


class PoseModel(object):

    def __init__(self, **kwargs):
        self.session = kwarget('session', None, **kwargs)

        self.model_name = kwarget('model_name', None, **kwargs)
        self.model_config = kwarget('model_config', None, **kwargs)
        self.input_list = kwarget('input_list', None, **kwargs)
        self.outputs = kwarget('output_array', [], **kwargs)

        self.load_config()
        self.load_model()

    def load_config(self, **kwargs):
        NotImplementedError("load_config function not implemented in " + self.model_name)

    def load_model(self, **kwargs):
        NotImplementedError("load_model function not implemented in " + self.model_name)

    def run_model(self, verbose=False):
        self.outputs = []
        n_data = len(self.input_list)
        for data_index, input_data in enumerate(self.input_list):
            if verbose:
                print("Processing input {}/{} for model {}".format(data_index + 1, n_data, self.model_name))
            self.outputs.append(self.run_model_once(input_data))

    def run_model_once(self, input_path):
        assert input_path
        NotImplementedError("run_model_once not implemented in " + self.model_name)

    def calculate_confidence(self):
        output_confidences = []
        for an_output in self.outputs:
            output_confidences.append(self.calculate_confidence_once(an_output))
        return output_confidences

    def calculate_confidence_once(self, input_path):
        NotImplementedError("calculate_confidence function not implemented in " + self.model_name)

    def make_heatmaps_once(self, an_output):
        NotImplementedError("make_heatmaps_once not implemented in " + self.model_name)


class HumanPoseModel(PoseModel):

    def __init__(self, **kwargs):
        self.internal_inputs = None
        self.internal_outputs = None
        super().__init__(model_name='HumanPose', **kwargs)
        self.joint_names = self.model_config.all_joints_names

    def load_config(self, config_path='./HumanPose/config.yaml'):
        cfg = default.cfg
        with open(config_path, 'r') as f:
            yaml_config = EasyDict(yaml.load(f))
        merge_a_to_b(yaml_config, cfg)
        self.model_config = cfg

    def load_model(self):
        self.session, self.internal_inputs, self.internal_outputs = setup_pose_prediction(self.model_config)

    def run_model_once(self, input_path):
        image = imread(input_path)
        image_batch = data_to_input(image)
        output = self.session.run(self.internal_outputs, feed_dict={self.internal_inputs: image_batch})
        scmap, locref, _ = extract_cnn_output(output, self.model_config)
        return [scmap, locref]

    def calculate_confidence_once(self, an_output, threshold=0.1):
        # Get joints processed from id
        all_joints = self.model_config.all_joints
        confidences = []
        for part in all_joints:
            # calculate resulting map for this joint
            scmap = an_output[0]
            scmap_part = np.sum(scmap[:, :, part], axis=2)

            # TODO: Try out max or mean data, which one to use?
            data = scmap_part.flatten()
            data = [a_value for a_value in data if a_value >= threshold]
            confidences.append(np.mean(data))
        return confidences

    def make_heatmaps_once(self, an_output):
        heatmaps = []
        for part in self.model_config.all_joints:
            scmap = an_output[0]
            scmap_part = np.sum(scmap[:, :, part], axis=2)
            # resize heatmap, it's eight times smaller
            scmap_part = imresize(scmap_part, 8.0, interp='bicubic')
            scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')
            heatmaps.append(scmap_part)
        return heatmaps


class PoseEstimationModel(PoseModel):

    def __init__(self, **kwargs):

        name = 'PoseEstimation'
        self.estimator = None
        self.target_size = None
        super().__init__(model_name=name, **kwargs)
        # TODO: resolve joint names
        self.generate_joint_list()

    def generate_joint_list(self):
        # TODO: check tf_pose/common.py for Coco vs MPII parts
        # TODO: changed this list, have to change usage inside class!
        # ankle, knee, hip, wrist, elbow, shoulder, chin, forehead
        joint_list = {(13, 10): 0, (12, 9): 1, (11, 8): 2, (7, 4): 3, (6, 3): 4, (5, 2): 5, (1, ): 6, (15, 14): 7}
        # joint_list = {13: 0, 12: 1, 11: 2, 7: 3, 6: 4, 5: 5, 1: 6, 15: 7}
        self.model_config.all_joints_list = joint_list

    def load_config(self, **kwargs):

        description = kwarget('description', 'tf-pose-estimation run', **kwargs)
        config = argparse.ArgumentParser(description=description)

        image_path = kwarget('image_path', '', **kwargs)
        config.add_argument('--image', type=str, default=image_path)

        model_type = kwarget('model_type', 'cmu', **kwargs)
        config.add_argument('--model', type=str, default=model_type, help='use "cmu" or "mobilenet_thin"')

        resizing = kwarget('resize', '0x0', **kwargs)
        config.add_argument('--resize', type=str, default=resizing,
                            help='if provided, resize images before they are processed. default=0x0, '
                                 'Recommends : 432x368 or 656x368 or 1312x736')
        resize_ratio = kwarget('ratio', 4.0, **kwargs)
        config.add_argument('--resize-out-ratio', type=float, default=resize_ratio,
                            help='if provided, resize heatmaps before they are post-processed. default=4.0')
        self.model_config = config.parse_args()

    def load_model(self, **kwargs):
        self.target_size = model_wh(self.model_config.resize)  # w, h
        if self.target_size[0] == 0 or self.target_size[1] == 0:
            target_size = (432, 368)
        else:
            target_size = self.target_size
        self.estimator = TfPoseEstimator(get_graph_path(self.model_config.model), target_size=target_size)

    def run_model_once(self, input_path):
        # TODO: This read_img is a bad function to use (scipy image will be deprecated)
        image = common.read_imgfile(input_path, None, None)
        humans = self.estimator.inference(image,
                                          resize_to_default=(self.target_size[0] > 0 and self.target_size[1] > 0),
                                          upsample_size=self.model_config.resize_out_ratio)
        if not humans:
            return None

        # TODO: We store only one human, must generalize for multiperson
        human = humans[0]
        human.heatmaps = self.estimator.heatMat
        return human

    def calculate_confidence_once(self, an_output):
        if an_output is None:  # we return nans
            confidence = np.empty(len(self.model_config.all_joints_list))
            confidence[:] = np.nan
            return confidence

        # if there's a human we return their confidences
        human_confidence = np.empty(len(self.model_config.all_joints_list))
        human_confidence[:] = 0
        for a_part in an_output.body_parts.items():
            real_indexes = [values for keys, values in self.model_config.all_joints_list.items() if a_part[0] in keys]
            for an_index in real_indexes:
                best_confidence = max(a_part[1].score, human_confidence[an_index])
                if human_confidence[an_index] > 0 and best_confidence == a_part[1].score:  # we're storing R side
                    human_confidence[an_index] = - best_confidence
                else:
                    human_confidence[an_index] = best_confidence  # L side (comes first and R last)
        # TODO: we store the first human found in the picture, should generalize that in the future for multiperson
        return human_confidence

    def make_heatmaps_once(self, an_output):
        map_dimensions = an_output.heatmaps[:, :, 0].shape
        map_dimensions = [2 * a_dim for a_dim in map_dimensions]  # heatmaps are half the original size
        heatmaps = [np.zeros(map_dimensions) for _ in range(len(self.model_config.all_joints_list.values()))]
        for a_part in an_output.body_parts.keys():
            real_indexes = [values for keys, values in self.model_config.all_joints_list.items() if a_part in keys]
            for an_index in real_indexes:
                heatmap = an_output.heatmaps[:, :, a_part]
                # TODO: Check correct dimensions (important or not?)
                heatmap = imresize(heatmap, 2.0, interp='bicubic')
                heatmaps[an_index] = heatmaps[an_index] + heatmap  # we add left + right heatmaps
        return heatmaps


# A stupid HumanPose function to merge dicts
def merge_a_to_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                merge_a_to_b(a[k], b[k])
            except ValueError:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
