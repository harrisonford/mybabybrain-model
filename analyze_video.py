import argparse
import tensorflow as tf
from poseModels import HumanPoseModel
import cv2
import json
import numpy as np
import pandas as pd
import pickle


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
    return pd.DataFrame(dummy_intervals, columns=['T0', 'Tf'])


def classify_subject_confidences(my_pickle_model, _confidence_list, _frame_list,
                                 window_size_seconds=10, fps=240, fskip=30, model_type=None):
    if not my_pickle_model:  # no model, we just use a dummy
        return classify_subject_dummy(np.max(_frame_list))

    window_size_data = int(window_size_seconds * fps / fskip)  # at 240fps and 30 fskip we have 8 data per second

    # calculate predictions
    # first we remove possible nans from data
    clean_confidence_list = [a_sample for a_sample in _confidence_list if not np.any(np.isnan(a_sample))]
    predictions = my_pickle_model.predict(clean_confidence_list)

    # compress into abnormal or gm windows
    # we will store the windows into a simple format t0, tf for desired windows
    compressed_windows = []
    for index in range(0, len(predictions), window_size_data):
        sub_sample = predictions[index:index + window_size_data - 1]
        score = np.nanmean(sub_sample)
        if (model_type == 'normal' and score < 0.5) or (model_type == 'gm' and score > 0.5):
            t0 = _frame_list[index] / fps
            t1 = _frame_list[index] / fps + window_size_seconds
            compressed_windows.append([t0, t1])

    # now we extend windows they're next to each other
    compressed_windows_extended = []
    stored_t0 = None
    stored_t1 = None
    if compressed_windows:
        stored_t0, stored_t1 = compressed_windows[0]

    for index, (t0, t1) in enumerate(compressed_windows):
        # we store the values if not connected to next window or last window
        if stored_t1 - t0 > window_size_seconds or index == len(compressed_windows) - 1:
            compressed_windows_extended.append([stored_t0, stored_t1])
            stored_t0 = t0
        else:  # we keep compressing windows
            stored_t1 = t1

    # now we make a DataFrame to save T0 and T1 and return it
    return pd.DataFrame(compressed_windows_extended, columns=['T0', 'Tf'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", type=str, default=None, help="full path to input video file")
    ap.add_argument("--out", type=str, default=None, help="full path to output video file")
    ap.add_argument("--fskip", type=int, default=30, help="set frame jump for each processed")
    args = vars(ap.parse_args())

    file_in = args['in']
    file_out = args['out']
    fskip = args['fskip']

    # prepare models
    tf.reset_default_graph()
    tracking_model = HumanPoseModel()
    abnormal_model = pickle.load(open('./ClassModels/classification_model_abnormal.pkl', 'rb'))
    gm_model = pickle.load(open('./ClassModels/classification_model_gm.pkl', 'rb'))

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
            continue

        this_time = video_capture.get(0)  # gets time in ms
        print("processing frame: {}; {}[s]".format(a_frame, this_time/1000))
        model_output = tracking_model._run_model_once(an_image)
        confidence_output = tracking_model.calculate_confidence_once(model_output)
        confidences.append(confidence_output)

    # prepare to save in a json file
    frame_list = np.array(processed_frames).tolist()
    confidence_list = np.array(confidences).tolist()
    data = dict(file_name=file_in, model_name=tracking_model.model_name, processed_frames=frame_list,
                confidences=confidence_list)
    # save the model data just in case
    # save_as_json(data, file_out + '.json')

    # classify subject data
    intervals_gm = classify_subject_confidences(gm_model, confidence_list, frame_list, model_type='gm')
    intervals_abnormal = classify_subject_confidences(abnormal_model, confidence_list, frame_list, model_type='normal')
    intervals_gm.to_csv(file_out + '_gm.csv')
    intervals_abnormal.to_csv(file_out + '_abnormal.csv')

    # print a message with the proportion of data marked as positive
    deltas_gm = intervals_gm['Tf'] - intervals_gm['T0']
    deltas_abnormal = intervals_abnormal['Tf'] - intervals_abnormal['T0']
    ratio_gm = np.sum(deltas_gm)/frame_list[-1]
    ratio_abnormal = np.sum(deltas_abnormal)/frame_list[-1]

    print("{}% of classified data as General Movement".format(100*ratio_gm))
    if ratio_gm == 0:  # we return to the dummy model, sorry Francisco :'(
        Warning('Because GM cannot be detected system will return to dummy model')
        intervals_gm = classify_subject_dummy(frame_list[-1])
        intervals_gm.to_csv(file_out + '_gm.csv')

    print("{}% of classified data as Abnormal".format(100 * ratio_abnormal))
    if 0 < ratio_abnormal < 0.1:
        print("Partially abnormal subject")
    elif ratio_abnormal >= 0.1:
        print("Highly abnormal subject")
    else:
        print("Normal subject")
