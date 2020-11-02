from yolov4.tf import YOLOv4
import cv2 as cv
import os , datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# initializing the model
yolo = YOLOv4()
yolo.classes = 'classes.names'
yolo.make_model()
yolo.load_weights("custom.weights", weights_type="yolo")

# getting the picture
img = cv.imread('plates/car.jpg')

_ = yolo.predict(img,score_threshold=0.8)      # predicting in first time is slower so I added this

# predicting the encoded bounding boxes
inp = input("ready to predict, press any key...")
start_time = datetime.datetime.now() # to estimate model speed
bboxes = yolo.predict(img,score_threshold=0.8)
end_time = datetime.datetime.now()      # to estimate model speed

# draw the bounding boxes nad save the picture
pred = yolo.draw_bboxes(img,bboxes)
cv.imwrite('pred.jpg',pred)

# get the bounding boxes for recognition part in future...
height, width, _ = img.shape
bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
for box in bboxes :
    w = int(box[2])
    h = int(box[3])
    x = int(box[0]) - int(w / 2)
    y = int(box[1]) - int(h / 2)
    plate = img[y:y + h, x:x + w]
    # cv.imshow('', plate)
    # cv.waitKey(0)

# estimating the model speed
timedelta = end_time - start_time
timedelta = str(timedelta).split(":")[2]
print(f'predicted in: {timedelta} seconds')

