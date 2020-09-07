import cv2 as cv
import cv2
import numpy as np
import math, os , pickle
from scipy import ndimage


def img_init(path,thresh=100):   # Initializes the image and does some process to make plate image ready for segmentation
    image = cv.imread(path)
    try:
        rimage = auto_rotate(image)  # tries to rotate image, if it is already in the right angle, gives error; so it's
    except:                          # in a try except statement
        rimage = image.copy()
    _, _, imgValue = cv.split(cv.cvtColor(rimage, cv.COLOR_RGB2HSV))   # lines below are some image processings
    structuringElement = np.ones((3, 3), np.uint8)
    imgTopHat = cv.morphologyEx(imgValue, cv.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv.morphologyEx(imgValue, cv.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv.add(imgValue, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    imgBlurred = cv.GaussianBlur(imgGrayscalePlusTopHatMinusBlackHat, (5, 5), 0)
    r, imgThresh = cv.threshold(imgBlurred, thresh, 255, cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    dilated = cv.dilate(imgThresh, kernel, iterations=1)
    return dilated


def auto_rotate(img):  # Gets an image and returns it in the right angle
    img_canny = cv.Canny(img, 100, 255, apertureSize=3)
    lines = cv.HoughLinesP(img_canny, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    img = ndimage.rotate(img, median_angle)
    return img


def plate_segmenter(img,path,thresh=50):    # Gets the plate and segments it to its characters, the returns an array of
                                            # character images
    image = cv.imread(path)
    try:
        image = auto_rotate(image)
    except:
        pass
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) # gets coordinates of detected
    height, width = img.shape                                                      # objects which could possibly be
                                                                                   # chareacters
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    contours = character_validation(contours,height,width)      #gets the contours which are characters
    d = 0
    characters = []
    for ctr in contours :
        x, y, w, h = cv.boundingRect(ctr)
        try:
            roi = img[y - 10:y + h + 10, x - 5:x + w + 2]  # makes a rectangled image of character with some blank spaces
            roi = cv.resize(roi, (30, 30))
        except:
            continue

        characters.append(roi)
        d += 1
    if len(characters) != 8 and thresh < 250:  # if the characters are invalid processes image again with upper
        img = img_init(path,thresh+5)
        print(len(characters),'   ',thresh)# thershold and segments the plate again
        # characters= plate_segmenter(img,path,thresh+5)

    else:
        thresh = 80
    characters = np.array(characters)
    return characters


def character_validation(contours,height,width): # checks if given contour is a character
    centerX = width / 2
    centerY = height / 2
    char_cnts = []
    for i, ctr in enumerate(contours):
        # Get bounding box
        x, y, w, h = cv.boundingRect(ctr)
        ratio = h / w
        cntX = x + (w / 2)
        cntY = (h / 2) + y
        if cntX < width / 8: # for contours which are in blue are of plate
            print('1')
            continue
        if math.sqrt((cntY - centerY) ** 2) > centerY / 2:
            print('2')# for some irrelevant contours which are upper or
            continue                                        # lower than characters
        if h > height * 0.8 or w > width / 6:       # for big irrelevant rectangles or a big part of plate
            print('3')
            continue
        if h < 10 and w < 10:  # characters should be bigger that size
            print('4')
            continue
        if h * w < 200:  # character 'zero' may be smaller than this size, that's for this character
            if math.sqrt((cntY - centerY) ** 2) > 10:
                print('5')
                continue
        if ratio > 3:   # for irrelevant rectangles
            print(6)
            continue
        try:            # if distance with previous found charcter is too low ignores it
            # some character like '5' have blank part inside, and findcontour detects it too, this condition
            # helps to get rid of second one
            x2, y2, w2, h2 = cv.boundingRect(char_cnts[-1])
            y_dist = (cntY - (y2 + h2 / 2)) ** 2
            x_dist = (cntX - (x2 + w2 / 2)) ** 2
            dist = math.sqrt(x_dist + y_dist)
            if dist < 25:
                continue
            # print(dist, cntY, w * h)
        except:
            pass

        char_cnts.append(ctr)
    return char_cnts


def create_datalist(path): # reads the path and returns a shuffled array of image names in the path
    datalist = np.array([])
    for name in os.listdir(path):
        datalist = np.append(datalist, name)
    np.random.shuffle(datalist)
    return datalist


def predict_train_split(datalist): # splits last 200 images for predicting
    train = datalist[:-200]
    pred = datalist[-200:]
    return train , pred


def label_set(label):  # given the image name, returns an array of corresponding characters
    labelset = np.array([label[0]])
    labelset = np.append(labelset, [label[1]])
    labelset = np.append(labelset, [label[2:-5]])
    labelset = np.append(labelset, [label[-5]])
    labelset = np.append(labelset, [label[-4]])
    labelset = np.append(labelset, [label[-3]])
    labelset = np.append(labelset, [label[-2]])
    labelset = np.append(labelset, [label[-1]])
    return labelset


def train_make(train_list, folderpath):
    # gets the list of images, extracts characters from images and writes them into train folder
    # data = np.array([np.zeros((30,30))]) # needed for concatenating, useless, will get rid of it some lines below
    i = -1 # for making every image name unique
    for name in train_list:
        perc = (i / len(train_list)) * 100
        path = folderpath + str(name)
        i += 1
        try:
            img = img_init(path)
            if perc % 5 == 0:
                print("segmenting {0}... \n *************************".format(perc))
            characters = plate_segmenter(img,path)
        except Exception as e:
            print(e)
            continue
        if len(characters) != 8 : # if number of characters is not 8 ignores that plate
            print("invalid chars")
            continue
        labels = np.array([])
        label = str(name).split('_')[0] # text after '_' is useless
        labelset = label_set(label)
        for j in range(8):
            filepath = "dataset/" +str( labelset[j]) + "_" + str(i) + ".png"
            if perc % 5 == 0:
                print("Writing files...     {0}% \n *************************".format(perc))
            # np.concatenate((data, [characters[j]]), axis=0)
            cv.imwrite(filepath, characters[j]) # writes the image into given folder
    # data = np.delete(data,0,0)   # we get rid of it here
    # with open('data.pickle','wb') as f :
    #     pickle.dump(data,f)       # for some unknown reason, data is empty here, so I commented it out


def data_init(): # a function to read, process and write data into the train folder
    datalist = create_datalist("roya_bold/")
    train, pred = predict_train_split(datalist)
    train_make(train, 'roya_bold/')

    with open("predict_list.txt", "w") as predictfile: # write predict image names into a text
        for name in pred:
            predictfile.write(name + "\n")
    print("\n\n\ndone")


def write_data():       # reads images from train folder and writes them into a pickle file to use in the model
    i = 0
    data = np.array([np.zeros((30,30,3))]) # needed for concatenating, useless, will get rid of it some lines below
    labels = np.array([])

    for name in os.listdir("dataset/"):
        perc = (i * 100 / 43991)
        path = "dataset/" + name
        img = cv.imread(path) # reads image in grayscale mode
        img = np.array(img)
        # print(img.shape)# makes it a numpy array
        data = np.concatenate((data,[img]),axis=0)
        labels = np.append(labels,str(name.split('_')[0]))
        print("****",name,"                   ",perc,'%')
        i +=1

    data = np.delete(data,0,0) # remember I said i will get rid of it? here we go
    # Writing data and labels into the corresponding pickle files
    with open("data.pickle",'wb') as f:
        pickle.dump(data,f)
    with open("labels.pickle",'wb') as f:
        pickle.dump(labels,f)


# write_data()
# data_init()
# path = 'roya_bold/67JIM39621_base35.png'
# img = img_init(path,80)
# characters = plate_segmenter(img,path)
# cv.imshow('',img)
# cv.waitKey(0)
# for char in characters :
#     cv.imshow('',char)
#     cv.waitKey(0)
