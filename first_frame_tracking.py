import os
from typing import List

from poseModels import PoseEstimationModel
import pandas as pd


if __name__ == '__main__':
    folder_list = ['/home/ssur-rtx01/Desktop/Dataset_Frames_CSV/images/train/False_High',
                   '/home/ssur-rtx01/Desktop/Dataset_Frames_CSV/images/train/False_Low',
                   '/home/ssur-rtx01/Desktop/Dataset_Frames_CSV/images/train/True_high',
                   '/home/ssur-rtx01/Desktop/Dataset_Frames_CSV/images/train/True_Low']

    model = PoseEstimationModel()
    total_data = []
    columns = ["file", "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
               "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee",
               "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]

    for a_folder in folder_list:
        image_list = sorted(os.listdir(a_folder))
        for an_image in image_list:
            print("Computing: {}".format(an_image))
            data = [None for _ in range(len(columns))]  # type: List[object]
            data[0] = a_folder + an_image  # file directory

            baby = model.run_model_once("{dir}/{frame}".format(dir=a_folder, frame=an_image))
            if baby is not None:
                parts = baby.body_parts
                for index, a_part in parts.items():
                    data[index + 1] = (a_part.x, a_part.y)
            total_data.append(data)

    df = pd.DataFrame(total_data, columns=columns)
    df.to_csv('tracking.csv')
