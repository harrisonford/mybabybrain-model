from qol import load_annotations, times_to_frames, get_frames_from, compute_error
from poseModels import PoseEstimationModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == '__main__':
    # read annotations
    annotation_folder = '/home/harrisonford/Downloads/'
    video_list_normal = ['000345']
    video_list_abnormal = ['000845']
    to_load_list_normal = [annotation_folder + an_id + '_coco.json' for an_id in video_list_normal]
    to_load_list_abnormal = [annotation_folder + an_id + '_coco.json' for an_id in video_list_abnormal]
    annotations_normal = [load_annotations(annotation) for annotation in to_load_list_normal]
    annotations_abnormal = [load_annotations(annotation) for annotation in to_load_list_abnormal]

    # extract times (id) to load from annotations
    image_ids_normal = [[image_dict['id'] for image_dict in annotation['images']]
                        for annotation in annotations_normal]
    image_ids_abnormal = [[image_dict['id'] for image_dict in annotation['images']]
                          for annotation in annotations_abnormal]
    # transform each time id to corresponding frame
    frame_ids_normal = [times_to_frames(ids) for ids in image_ids_normal]
    frame_ids_abnormal = [times_to_frames(ids) for ids in image_ids_abnormal]

    # extract image frames from videos to process
    # TODO: frame id from getvalue is not perfectly correct
    video_folder = '/home/harrisonford/Videos/babybrain-converted/'
    image_container = [get_frames_from(video_folder + video + '.MP4', frame_list, threshold=1) for (video, frame_list)
                       in zip(video_list_normal + video_list_abnormal, frame_ids_normal + frame_ids_abnormal)]

    # calculate tracking for each image in container
    pose_model = PoseEstimationModel()
    id_list = []
    for a_video_container, video_ids in zip(image_container, image_ids_normal + image_ids_abnormal):
        for an_image, an_id in zip(a_video_container, video_ids):
            pose_model.outputs.append(pose_model._run_model_once(an_image))
            id_list.append(an_id)

    # save tracking in coco format
    output_json = '/home/harrisonford/Downloads/video_pose_model_coco.json'
    pose_model.save_as_coco_result(output_json, id_vector=id_list)

    # calculate curve performance
    distances_normal, visible_normal = \
        compute_error(output_json, annotation_folder + video_list_normal[0] + '_coco.json', normalize='body')

    distances_abnormal, visible_abnormal = \
        compute_error(output_json, annotation_folder + video_list_abnormal[0] + '_coco.json', normalize='body')

    # for each detection threshold count the data that falls in, for each joint
    threshold = np.array(range(101))/100
    threshold_total_count_normal = []
    threshold_total_count_abnormal = []
    for a_threshold in threshold:  # count for each threshold
        threshold_count_normal = np.zeros(distances_normal[0].shape)
        threshold_count_abnormal = np.zeros(distances_abnormal[0].shape)
        for a_sample_distance in distances_normal:  # accumulate from all samples
            for index, a_joint_distance in enumerate(a_sample_distance):  # for each joint
                if a_joint_distance <= a_threshold:
                    threshold_count_normal[index] += 1
        for a_sample_distance in distances_abnormal:  # accumulate from all samples
            for index, a_joint_distance in enumerate(a_sample_distance):  # for each joint
                if a_joint_distance <= a_threshold:
                    threshold_count_abnormal[index] += 1
        threshold_total_count_normal.append(threshold_count_normal)
        threshold_total_count_abnormal.append(threshold_count_abnormal)

    # finally normalize by quantity
    threshold_total_count_normal = [count_sample / len(image_ids_normal[0])
                                    for count_sample in threshold_total_count_normal]
    threshold_total_count_abnormal = [count_sample / len(image_ids_abnormal[0])
                                      for count_sample in threshold_total_count_abnormal]
    # save in a data frame so it's easier to plot
    data = []
    for threshold_index, a_threshold in enumerate(threshold_total_count_normal):
        for joint_index, a_value in enumerate(a_threshold):
            data.append([threshold[threshold_index], joint_index, a_value, 'normal'])
    for threshold_index, a_threshold in enumerate(threshold_total_count_abnormal):
        for joint_index, a_value in enumerate(a_threshold):
            data.append([threshold[threshold_index], joint_index, a_value, 'abnormal'])
    detect_df = pd.DataFrame(data=data, columns=['threshold', 'joint', 'ratio', 'class'])
    sns.set()
    ax = sns.lineplot(x='threshold', y='ratio', data=detect_df, hue='class')
    plt.show()

    # now calculate COCO results
    coco_gt = COCO(to_load_list_normal[0])
    coco_dt = coco_gt.loadRes(output_json)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print('done!')
