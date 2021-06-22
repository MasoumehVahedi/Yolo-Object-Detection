
from yolo_utils import read_classes, read_anchors
from utils import predict_yolo_model
from tensorflow.keras.models import load_model



"""
   We need to convert Darknet YOLO_v2 model to the Keras model (yolov2.h5) for loading the pre-trained yolo model to use it.
   
   To use yad2k (or YOLO version 2), we need to follow these steps (in google colab):
       The yolo.h5 file can be generated using the YAD2K repository here: "https://github.com/allanzelener/YAD2K"
       
       Steps how to do it on windows:
       
       1- Clone the above repository to your computer (or colab): "git clone https://github.com/allanzelener/yad2k.git"
       
      * Note : After clone the repository, we must go inside the directory using os.chdir(folder_name):
             os.chdir("/content/yad2k/")
       
       2- Download the yolo.weights file from here : "http://pjreddie.com/media/files/yolov2.weights"
       
       3- Download the yolo.cfg file form here : "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg"
       
       4- Convert the Darknet YOLO_v2 model to the Keras model: "./yad2k.py yolov2.cfg yolov2.weights model_data/yolov2.h5"
       
   NOTE :
       There are some challenges with version of tensorflow (2.5.0) and keras to convert "yolov2.h5" file based on 4 phases above.
       So, to solve this problem, we should change **tf.space_to_depth(x, block_size=2)** to **tf.nn.space_to_depth(x, block_size=2)** in 
       "space_to_depth_x2" function in "keras_yolo.py".
       The directory is here: yad2k > yad2k > models > keras_yolo.py

"""

# git clone https://github.com/allanzelener/yad2k.git
# Go inside the directory using os.chdir(folder_name)
# os.chdir("/content/yad2k/")

# !wget http://pjreddie.com/media/files/yolov2.weights
# !wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg
# ! ./yad2k.py yolov2.cfg yolov2.weights model_data/yolov2.h5




if __name__ == "__main__":
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    # Same as yolo_model input layer size
    model_image_size = (608, 608)

    # Loading a Pre-trained Model
    yolo_model = load_model("/content/yad2k/model_data/yolov2.h5", compile=False)
    print(yolo_model.summary())

    out_scores, out_boxes, out_classes = predict_yolo_model(image_file="test.jpg",
                                                            anchors = anchors,
                                                            yolo_model=yolo_model,
                                                            class_names=class_names)


