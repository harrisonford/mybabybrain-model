from qol import kwarget
import yaml
from easydict import EasyDict
from HumanPose.nnet.predict import setup_pose_prediction, extract_cnn_output
from HumanPose.dataset.pose_dataset import data_to_input


class PoseModel(object):

    def __init__(self, **kwargs):
        self.inputs = kwarget('inputs', None, **kwargs)
        self.outputs = kwarget('outputs', None, **kwargs)
        self.session = kwarget('session', None, **kwargs)

        self.model_name = kwarget('model_name', None, **kwargs)
        self.model_config = kwarget('model_config', None, **kwargs)
        self.config_path = kwarget('config_path', None, **kwargs)

        self.load_config()
        self.load_model()

    def load_config(self):
        NotImplementedError("load_config function not implemented in " + self.model_name)

    def load_model(self):
        NotImplementedError("load_model function not implemented in " + self.model_name)

    def run_model(self):

        outputs = []
        for input_data in self.inputs:
            outputs.append(self.run_model_once(input_data))
        return outputs

    def run_model_once(self, input_data):
        assert input_data
        NotImplementedError("run_model_once not implemented in " + self.model_name)

    def calculate_confidence(self, **kwargs):
        assert kwargs
        NotImplementedError("calculate_confidence function not implemented in " + self.model_name)


class HumanPoseModel(PoseModel):

    def __init__(self):
        name = 'HumanPose'
        config = './HumanPose/config.yaml'
        super().__init__(model_name=name, config_path=config)

    def load_config(self):

        with open(self.config_path, 'r') as f:
            self.model_config = EasyDict(yaml.load(f))

    def load_model(self):
        self.session, self.inputs, self.outputs = setup_pose_prediction(self.model_config)

    def run_model_once(self, input_data):
        image_batch = data_to_input(input_data)
        output = self.session.run(self.outputs, feed_dict={self.inputs: image_batch})
        scmap, locref, _ = extract_cnn_output(output, self.model_config)
        return scmap, locref
