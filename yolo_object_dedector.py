import cv2
import numpy as np

class YOLOv3:

    def __init__(self):
        self.img = None
        self.model = None
        self.classNames = []
        self.output_layers = []

        self.initialize_network()

    def initialize_network(self):

        self.model = cv2.dnn.readNet("yolov3.cfg","yolov3.weights")
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        layers = self.model.getLayerNames()
        unconnect = self.model.getUnconnectedOutLayers()
        unconnect = unconnect-1    
        
        for i in unconnect:
            self.output_layers.append(layers[int(i)])
    
        classFile = 'coco.names'        
        with open(classFile,'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

    def detect(self, img):
    
        img_width = img.shape[1]
        img_height = img.shape[0]
    
        img_blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), swapRB=True)
    
        self.model.setInput(img_blob)
        detection_layers = self.model.forward(self.output_layers)
    
        ids_list = []
        boxes_list = []
        confidences_list = []
    
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]
    
                if confidence > 0.10:
    
                    label = self.classNames[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                    start_x = int(box_center_x-(box_width/2))
                    start_y = int(box_center_y-(box_height/2))
    
                    ids_list.append(predicted_id)
                    confidences_list.append(float(confidence))
                    boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
    
        max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    
        for max_id in max_ids:
            max_class_id = max_id
            box = boxes_list[int(max_class_id)]
            start_x = box[0]
            start_y = box[1]
            box_width = box[2]
            box_height = box[3]
    
            predicted_id = ids_list[int(max_class_id)]
            label = self.classNames[predicted_id]
            confidence = confidences_list[int(max_class_id)]
    
            end_x = start_x + box_width
            end_y = start_y + box_height
    
            cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(255, 0, 0),2)
    
            cv2.putText(img,label + " %" + str(confidence * 100),(start_x,start_y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),1,1)

        return img
