from imageai.Detection import VideoObjectDetection

import os

detector = VideoObjectDetection()

execution_path = os.getcwd()

detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel("flash") #
detector.loadModel()
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                            output_file_path=os.path.join(execution_path, "traffic_detected.mp4")
                            , frames_per_second=20, log_progress=True)
print(video_path)
