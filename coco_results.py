# import os
from performance_curves import load_annotations
import tensorflow as tf
from poseModels import HumanPoseModel, PoseEstimationModel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoanalyze import COCOanalyze


if __name__ == '__main__':
    annotation_path = '/home/babybrain/Escritorio/000845_coco.json'
    frames_path = '/home/babybrain/Escritorio/000845/'
    outpath = '/home/babybrain/Escritorio/'

    fake_gt = load_annotations('/home/babybrain/Escritorio/person_keypoints_val2014.json')
    fake_dt = load_annotations('/home/babybrain/Escritorio/person_keypoints_val2014_fakekeypoints100_results.json')

    my_gt = load_annotations('/home/babybrain/Escritorio/000845_coco.json')
    my_dt = load_annotations('/home/babybrain/Escritorio/000845_pose.json')

    subsample = 120
    n_stop = 602
    # file_names = sorted(os.listdir(frames_path))
    # file_names = [a_frame for num, a_frame in enumerate(file_names) if num % subsample == 0 and num <= n_stop]
    file_names = ['000845_{:05d}.png'.format(num) for num in range(1, n_stop, subsample)]
    full_paths = [frames_path + a_frame for a_frame in file_names]

    # run models
    # model_pose = PoseEstimationModel(input_list=full_paths)
    # model_pose.run_model(verbose=True)
    # model_pose.save_as_coco_result(outpath + '000845_pose.json')

    # tf.reset_default_graph()
    # model_human = HumanPoseModel(input_list=full_paths)
    # model_human.run_model(verbose=True)
    # model_human.save_as_coco_result(outpath + '000845_human.json')

    # TODO: test pycocotools and the two files it uses: groundtruth and results (from save_as_coco_result)
    coco_groundtruth = COCO(annotation_path)
    coco_result_pose = coco_groundtruth.loadRes(outpath + '000845_pose.json')
    # coco_groundtruth = COCO('/home/babybrain/Escritorio/person_keypoints_val2014.json')
    # coco_result_pose = \
    #     coco_groundtruth.loadRes('/home/babybrain/Escritorio/person_keypoints_val2014_fakekeypoints100_results.json')

    # sub-image group
    # imgIds = sorted(coco_groundtruth.getImgIds())[0:100]
    imgIds = sorted(coco_groundtruth.getImgIds())

    # coco og eval
    # coco_eval = COCOeval(coco_groundtruth, coco_result_pose, 'keypoints')
    # coco_eval.params.imgIds = imgIds
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    # coco analyze
    coco_analysis = COCOanalyze(coco_groundtruth, coco_result_pose, 'keypoints')
    coco_analysis.cocoEval.params.imgIds = imgIds
    coco_analysis.evaluate(verbose=True, makeplots=True)
