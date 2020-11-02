**yolo object detection**

- to detect plates, open yolo.py and change the path to your desired picture. Then run it, and the predicted bounding boxes will be saved as pred.jpg in this directory.
- I tried to train this with my own dataset, but since my dataset was not enough, and there was no ready dataset in Iran, I used a pre trained Weights file from [theAIGuysCode](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) and implemented it with my own code.
- if the model did not predict well, try to reduce the score threshold since I increased it to 0.8.
- The link to [custom.weights file](https://drive.google.com/file/d/1EUPtbtdF0bjRtNjGv436vDY28EN5DXDH/view?usp=sharing). rename it to 'custom.weights' 
