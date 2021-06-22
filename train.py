import os

from tensorflow.keras.models import load_model
from model import YOLOv3, WeightReader

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims

from utils import decode_netout
from utils import correct_yolo_boxes
from utils import do_nms
from utils import get_boxes
from utils import draw_boxes



# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


if __name__ == "__main__":

    ###### Create YOLOv3 ######
    # Define the model
    model = YOLOv3()
    # Load the model weights
    weight_reader = WeightReader('D:/lyft/yolov3.weights')
    # Set the model weigths
    weight_reader.load_weights(model)
    # Save the model
    model.save("D:/lyft/model_yolov3.h5")

    # Load YOLOv3 Model
    yolo_model = load_model("D:/lyft/model_yolov3.h5")
    print(yolo_model.summary())

    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    # Set the input layer size
    IMG_WIDTH = 416
    IMG_HEIGHT = 416

    # Set the probability threshold to detect object
    # if [ highest class probability score < threshold], then get rid of the corresponding box
    score_threshold = 0.8

    ########## load images #########
    # Only detect objects from first 20 training images
    path = ".../lyft"
    images_dir = os.listdir(".../lyft/train_images")[:20]

    # Now create a loop to iterate images and find objects
    for file in images_dir:
        img_filename = path + "/train_images/" + file

        # load and prepare an image
        image, width, height = load_image_pixels(img_filename, (IMG_WIDTH, IMG_HEIGHT))

        # Predict images
        y_pred = yolo_model.predict(image)

        # Create boxes
        boxes = list()
        for i in range(len(y_pred)):
            # decode the output of the network
            boxes += decode_netout(y_pred[i][0], anchors[i], score_threshold, IMG_HEIGHT, IMG_WIDTH)

        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, height, width, IMG_HEIGHT, IMG_WIDTH)

        # suppress non-maximal boxe
        do_nms(boxes, 0.5)

        # define the labels (I chose those one that are relevant for this dataset, which were used in pretraining the YOLOv3 model)
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck","boat"]

        # get the details of the detected objects
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, score_threshold)

        # summarize what we found
        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        # draw what we found
        draw_boxes(img_filename, v_boxes, v_labels, v_scores)




