import argparse
import tensorflow as tf
from poseModels import HumanPoseModel
import cv2
import json
import numpy as np
import pandas as pd


def save_as_json(serializable_data, file_path):
    output = json.dumps(serializable_data)
    with open(file_path, "w") as f:
        f.write(output)


def classify_subject_dummy(total_frames):
    total_duration = total_frames/240
    # generate two intervals
    starts = [total_duration*0.15, total_duration*0.5]
    durations = [total_duration*0.1, total_duration*0.15]
    dummy_intervals = [[starts[0], starts[0]+durations[0]],
                       [starts[1], starts[1]+durations[1]]]
    return dummy_intervals


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", type=str, default=None, help="full path to input video file")
    ap.add_argument("--out", type=str, default=None, help="full path to output video file")
    ap.add_argument("--fskip", type=int, default=30, help="set frame jump for each processed")
    args = vars(ap.parse_args())

    file_in = args['in']
    file_out = args['out']
    fskip = args['fskip']

    # prepare model
    tf.reset_default_graph()
    model = HumanPoseModel()

    # container to save confidence values
    confidences = []

    # process per frame
    video_capture = cv2.VideoCapture(file_in)

    # get video frame count and list frames to process
    frame_count = int(video_capture.get(7))  # 7 = frame count
    processed_frames = [frame for frame in range(0, frame_count, fskip)]

    for a_frame in processed_frames:
        # set frame position to capture
        video_capture.set(1, a_frame)  # frame reference is set as 0-count
        success, an_image = video_capture.read()

        if not success:
            Warning("{} video file failed to process frame {}".format(file_in, a_frame))
            break

        this_time = video_capture.get(0)  # gets time in ms
        print("processing frame: {}; {}[s]".format(a_frame, this_time/1000))
        model_output = model._run_model_once(an_image)
        confidence_output = model.calculate_confidence_once(model_output)
        confidences.append(confidence_output)

    # prepare to save in a json file
    frame_list = np.array(processed_frames).tolist()
    confidence_list = np.array(confidences).tolist()
    data = dict(file_name=file_in, model_name=model.model_name, processed_frames=frame_list,
                confidences=confidence_list)
    # TODO: Working on final step classification will attach a dummy model for now
    # save_as_json(data, file_out)

    # classify subject data
    intervals = classify_subject_dummy(frame_count)

    # save intervals as DataFrame
    df = pd.DataFrame(intervals, columns=['T0', 'Tf'])
    df.to_csv(file_out)
