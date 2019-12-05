import argparse
import tensorflow as tf
from poseModels import HumanPoseModel
import cv2
import json


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", type=str, default=None, help="full path to input video file")
    ap.add_argument("--out", type=str, default=None, help="full path to output video file")
    ap.add_argument("--fps", type=int, default=30, help="set frames per second processed")
    args = vars(ap.parse_args())

    file_in = args['in']
    file_out = args['out']
    fps = args['fps']

    # prepare model
    tf.reset_default_graph()
    model = HumanPoseModel()

    # container to save confidence values
    confidences = []
    processed_frames = []

    # process per frame
    video_capture = cv2.VideoCapture(file_in)
    success, an_image = video_capture.read()
    last_frame = -fps
    while success:
        this_frame = video_capture.get(0)
        if this_frame - last_frame >= 1/fps:  # process the frame
            print("processing frame: {}".format(this_frame))
            model_output = model._run_model_once(an_image)
            confidence_output = model.calculate_confidence_once(model_output)
            confidences.append(confidence_output)
            processed_frames.append(this_frame)

            success, image = video_capture.read()
            last_frame = this_frame

    # prepare to save in a json file
    data = dict(file_name=file_in, model_name=model.model_name, processed_frames=processed_frames,
                confidences=confidences)
    json.dump(data, file_out)
