import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from yolo_utils import get_colors_for_classes, preprocess_image, draw_boxes, yolo_head
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import load_model

# git clone https://github.com/allanzelener/yad2k.git
from yolo_utils import scale_boxes



# Step 1 : Filters YOLO boxes by thresholding on object and class confidence.

def filter_yolo_boxes(boxes, box_confidence, prob_box_class, threshold):
    """
    param boxes: the shape of boxes is (19, 19, 5, 4)
    param box_confidence: the shape of box_confidence is (19, 19, 5, 1)
    param prob_box_class: the shape of prob_box_class is (19, 19, 5, 80)
    param threshold: we define a threshold to get rid of unnecessary boxes,
                     if the highest box_class score is less than threshold (highest box_class < threshold),
                     so we remove the corresponding box.

    return:
           This function will give  three components:
           scores: including the probability of class score for selected box, the shape is (None,)
           boxes: including the (b_x, b_y, b_h, b_w) coordinates of selected boxes, the shape is (None,4)
           classes: including the index of the class detected by the selected box, the shape is (None,)

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    box_scores = box_confidence * prob_box_class
    print("box_confidence = ", box_confidence.shape)
    print("prob_box_class = ", prob_box_class.shape)
    print("box_scores = ", box_scores.shape)
    # we need to determine the index of the classs with the maximum box_scores
    # in order to find the box_classes, then finding the corresponding box score
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_class_scores = tf.math.reduce_max(box_scores, axis = -1)
    filter_mask = box_class_scores >= threshold

    # Apply the mask to box_class_scores, boxes and box_classes
    scores = tf.boolean_mask(box_class_scores, filter_mask)
    boxes = tf.boolean_mask(boxes, filter_mask)
    classes = tf.boolean_mask(box_classes, filter_mask)

    return scores, boxes, classes

# Step 2 : A second filter for selecting the right boxes is called non-maximum suppression (NMS):
# Non-max suppression uses the very important function called "Intersection over Union", or IoU.
# so, implement the intersection over union (IoU) between box1 and box2
def IoU(box1, box2):
    """
    To calculate Intersection Over Union (IoU), will be done in the following steps:
    1- Calculate the (xi1, yi1, xi2, yi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    2- Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    3- Compute the IoU
    """

    (bx1_x1, bx1_y1, bx1_x2, bx1_y2) = box1
    (bx2_x1, bx2_y1, bx2_x2, bx2_y2) = box2

    # Step 1
    xi_1 = np.max([box1[0], box2[0]])
    yi_1 = np.max([box1[1], box2[1]])
    xi_2 = np.min([box1[2], box2[2]])
    yi_2 = np.min([box1[3], box2[3]])
    # max(height, 0) and max(width, 0)
    intersection_area = max((yi_2 - yi_1), 0) * max((xi_2 - xi_1), 0)

    # Step 2
    area_of_box1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    area_of_box2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_boxes = (area_of_box1 + area_of_box2) - intersection_area

    # Step 3
    iou = intersection_area / union_boxes

    return iou

#Step 3 : Now, we implement YOLO Non-max Suppression

""" There are three important phrase to implement YOLO Non-max Suppression:
         1 - First of all, we need to select the box with the high score
         2 - Then, the overlap of the highest score box will be compared with all other boxes
             to remove those that have (iou >= iou_threshold)
         3- Go back to step 1 and iterate it when finally there are no more boxes with a
            lower  score than the currently selected box.
"""
def YOLO_Non_max_Suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
       Arguments:
               scores -- tensor of shape (None,), output of yolo_filter_boxes()
               boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size
               classes -- tensor of shape (None,), output of yolo_filter_boxes()
               max_boxes -- integer, maximum number of predicted boxes you'd like
               iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

        Returns:
               scores -- tensor of shape (, None), predicted score for each box
               boxes -- tensor of shape (4, None), predicted box coordinates
               classes -- tensor of shape (, None), predicted class for each box

         Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
         function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    tensor_max_boxes = tf.Variable(max_boxes, dtype="int32")

    # Now, using  tf.image.non_max_suppression() to get the list of indices corresponding to boxes we keep
    indices_nms = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Uisng tf.gather() to select only nms_indices from scores, boxes and classes
    # tf.gather() is used to slice the input tensor based on the indices provided.
    scores = tf.gather(scores, indices_nms)
    boxes = tf.gather(boxes, indices_nms)
    classes = tf.gather(classes, indices_nms)

    return scores, boxes, classes


# There're a few ways of representing boxes, such as via their corners or
# via their midpoint and height/width which converts the yolo box coordinates (x,y,w,h) to
# box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes.
def yolo_box_coordinates_to_box_corner(xy_box, wh_box):
    min_box = xy_box - (wh_box / 2.)
    max_box = xy_box + (wh_box / 2.)

    return tf.keras.backend.concatenate([
        min_box[..., 1:2],  # y_min
        min_box[..., 0:1],  # x_min
        max_box[..., 1:2],  # y_max
        max_box[..., 0:1]  # x_max
    ])

# Step 4 : now, we need to implement a function to take the output of deep CNN which has 19x19x5x85
# dimensional encoding, and then filter through all the boxes using "YOLO_Non_max_Suppression" function.

######################## NOTE ##########################
# image_shape=(720., 1280.) --> has to be float32 dtype (if be int, we will give error)
def YOLO_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold = 0.6, iou_threshold = 0.5):
    """
       This function takes the output of the YOLO encoding and filters the boxes using score
       threshold and NMS.

       Arguments:
              yolo_outputs : output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                        xy_box: tensor of shape (None, 19, 19, 5, 2)
                        wh_box: tensor of shape (None, 19, 19, 5, 2)
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
              image_shape : tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
              max_boxes : integer, maximum number of predicted boxes you'd like
              score_threshold : real value, if [ highest class probability score < threshold], then get rid of the corresponding box
              iou_threshold : real value, "intersection over union" threshold used for NMS filtering

       Returns:
              scores : tensor of shape (None, ), predicted score for each box
              boxes : tensor of shape (None, 4), predicted box coordinates
              classes : tensor of shape (None,), predicted class for each box
    """
    xy_box, wh_box, box_confidence, prob_box_class = yolo_outputs

    # Convert boxes to be ready for filtering functions using yolo_box_coordinates_to_box_corner function
    boxes = yolo_box_coordinates_to_box_corner(xy_box, wh_box)

    # to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = filter_yolo_boxes(boxes, box_confidence, prob_box_class, threshold=score_threshold)

    # Scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)

    # Now, to perform Non-max suppression
    scores, boxes, classes = YOLO_Non_max_Suppression(scores, boxes, classes,
                                                      max_boxes=max_boxes,
                                                      iou_threshold=iou_threshold)
    
    return scores, boxes, classes


def predict_yolo_model(image_file, anchors, yolo_model, class_names):
    """
        Runs the graph to predict boxes for "image_file". Prints and plots the predictions.

        Arguments:
        image_file -- name of an image stored in the "images" folder.

        Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes

        Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
        """

    # Preprocessing the car detection images
    img, img_data = preprocess_image(image_file, model_image_size=(608, 608))

    yolo_model_outputs = yolo_model(img_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    out_scores, out_boxes, out_classes = YOLO_eval(yolo_outputs)

    # how many boxes find for the image
    print("Found {} boxes for {}".format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes
    colors = get_colors_for_classes(len(class_names))
    # Draw bounding boxes on the image file
    draw_boxes(img, out_boxes, out_classes, class_names, out_scores)
    # Save the predicted bounding box on the image
    img.save(os.path.join("/content/sample_data/out", image_file), quality=100)
    # cv2.imwrite("/content/sample_data/out" + image_file, img)
    # Display the results
    out_images = Image.open(os.path.join("/content/sample_data/out", image_file))
    plt.imshow(out_images)

    return out_scores, out_boxes, out_classes
