# TODO: YOLO usage should be super modular giving it an image from imageio and return coordinates
import argparse
import numpy as np
import cv2


# take an image read as a matrix using imageio and apply yolo model
def use_yolo(image_matrix):

    # load coco labels
    labels_path = './yolov3/data/coco.names'
    labels = open(labels_path).read().strip().split('\n')

    # list of random colors
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3))

    # load yolo weight and config
    weights_path = './yolov3/weights/yolov3-spp.weights'
    config_path = './yolov3/cfg/yolov3-spp.cfg'
    print('[INFO] Loading YOLO weights from {}'.format(weights_path))
    print('[INFO] Loading YOLO config file from {}'.format(config_path))
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return image_matrix


if __name__ == '__main__':
    use_yolo(None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="path to input file")
    args = vars(ap.parse_args())
