from qol import kwarget
import yaml
from easydict import EasyDict
from HumanPose.nnet.predict import setup_pose_prediction, extract_cnn_output
from HumanPose.dataset.pose_dataset import data_to_input
import HumanPose.default_config as default
from scipy.misc import imread
import numpy as np


class PoseModel(object):

    def __init__(self, **kwargs):

        self.session = kwarget('session', None, **kwargs)

        self.model_name = kwarget('model_name', None, **kwargs)
        self.model_config = kwarget('model_config', None, **kwargs)
        self.config_path = kwarget('config_path', None, **kwargs)
        self.input_list = kwarget('input_list', None, **kwargs)

        self.load_config()
        self.load_model()

    def load_config(self):
        NotImplementedError("load_config function not implemented in " + self.model_name)

    def load_model(self):
        NotImplementedError("load_model function not implemented in " + self.model_name)

    def run_model(self):

        outputs = []
        for input_data in self.input_list:
            outputs.append(self.run_model_once(input_data))
        return outputs

    def run_model_once(self, input_path):
        assert input_path
        NotImplementedError("run_model_once not implemented in " + self.model_name)

    def calculate_confidence(self, **kwargs):
        assert kwargs
        NotImplementedError("calculate_confidence function not implemented in " + self.model_name)


class HumanPoseModel(PoseModel):

    def __init__(self):
        name = 'HumanPose'
        config = './HumanPose/config.yaml'
        self.inputs = None
        self.outputs = None

        super().__init__(model_name=name, config_path=config)

    def load_config(self):

        cfg = default.cfg
        with open(self.config_path, 'r') as f:
            yaml_config = EasyDict(yaml.load(f))
        merge_a_to_b(yaml_config, cfg)
        self.model_config = cfg

    def load_model(self):
        self.session, self.inputs, self.outputs = setup_pose_prediction(self.model_config)

    def run_model_once(self, input_path):
        image = imread(input_path)
        image_batch = data_to_input(image)
        output = self.session.run(self.outputs, feed_dict={self.inputs: image_batch})
        scmap, locref, _ = extract_cnn_output(output, self.model_config)
        return scmap, locref

    def calculate_confidence(self, scmap, threshold=0.2):
        # Get joints processed from id
        all_joints = self.model_config.all_joints
        all_joints_names = self.model_config.all_joints_names

        confidences = []
        for pidx, part in enumerate(all_joints):
            # Calculate the result map for a joint (pre-heatmap)
            scmap_part = np.sum(scmap[:, :, part], axis=2)

            # For a part, we average every conf point higher than a threshold
            # TODO: Try out max or mean data, which one to use?
            data = [x for x in scmap_part.flatten() if x > threshold]
            confidences.append(np.max(data))  # using max to explore results
        return confidences, all_joints_names


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
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
