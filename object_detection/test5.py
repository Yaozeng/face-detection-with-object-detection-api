from PyQt5.QtWidgets import QApplication,QWidget,QFileDialog,QPushButton,QLabel,QLineEdit,QHBoxLayout,QVBoxLayout
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.ql1=QLabel('input',self)
        self.qle1=QLineEdit(self)
        self.pbtn1=QPushButton('choose file',self)
        self.pbtn1.clicked.connect(self.clicked1)


        self.ql2=QLabel('output',self)
        self.qle2 = QLineEdit(self)
        self.pbtn2 = QPushButton('choose file', self)
        self.pbtn2.clicked.connect(self.clicked2)


        self.pbtn3=QPushButton('run', self)
        self.pbtn3.clicked.connect(self.clicked3)

        self.hbox1=QHBoxLayout()
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.ql1)
        self.hbox1.addWidget(self.qle1)
        self.hbox1.addWidget(self.pbtn1)

        self.hbox2=QHBoxLayout()
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.ql2)
        self.hbox2.addWidget(self.qle2)
        self.hbox2.addWidget(self.pbtn2)

        self.vbox=QVBoxLayout()
        #vbox.addStretch(1)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addWidget(self.pbtn3)

        self.setLayout(self.vbox)
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('Buttons')
        self.show()
    def clicked1(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "打开文件", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.qle1.setText(fileName)
    def clicked2(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "保存文件", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.qle2.setText(fileName)
    def clicked3(self):
        self.run(self.qle1.text(),self.qle2.text())
    def run(self,source,des):
        # What model to download.
        MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        TEST_IMAGE_PATHS = source

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                videoCapture = cv2.VideoCapture(TEST_IMAGE_PATHS)
                size = (
                int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = videoCapture.get(cv2.CAP_PROP_FPS)
                videoWriter = cv2.VideoWriter(des, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
                success, frame = videoCapture.read()
                while success:
                    image_np_expanded = np.expand_dims(frame, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # image_np=image_np.transpose(1,0,2)
                    videoWriter.write(frame)
                    success, frame = videoCapture.read()
if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())



