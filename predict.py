from tensorflow.keras.models import  load_model
import numpy as np
import data_process
import cv2 as cv

# for converting encoded integer to string
alphabet_dict = { "0" : '0' , '1':'1', '2':'2' , '3':'3','4' :'4' ,'5' :'5' ,'6' :'6','7':'7' ,'8':'8' ,'9':'9' ,'10':'BE' ,
                '11':'CHE' ,'12':'DAL','13':'EIN' ,'14':'FE' ,'15':'GAF' ,'16':'GHAF' ,'17':'GHEIN' ,'18':'HA' ,'19':'HE' ,'20':'JIM','21' : 'KAF' ,
                  '22':'KHE','23' : 'LAM',
                  '24':'MIM' ,'25':'NON' ,'26':'PE' ,'27':'RE' ,'28':'SAD' ,'29':'SE' ,'30':'SHIN' ,'31':'SIN' ,'32':'TA' ,'33':'TE' ,
                    '34':'VAV' ,'35':'YE' ,'36':'ZA' ,'37':'ZAD' ,'38':'ZAL' ,'39':'ZE' ,'40':'ZHE'   }

# reads image and # gets characters out of it
path = 'cropp/17BE15643_48949.jpg'
img = data_process.img_init(path)
characters = data_process.plate_segmenter(img,path)
# characters = np.array(characters)
print(len(characters))
chars = []
# for i in range(6):
#     im = cv.cvtColor(characters[i],cv.COLOR_BGR2BGRA)
#     chars.append(im)
# chars = np.array(chars)
characters = characters.reshape(characters.shape[0],30,30,1)
# characters = chars.reshape(chars.shape[0],30,30,4)
# img1 = cv.imread(path)
# cv.imshow('',img)
# cv.waitKey(0)
for i in characters :
    cv.imshow('',i)
    cv.waitKey(0)

# loading model
model = load_model('cnn_classifier2.h5')

# predicting and getting output string
output = model.predict(characters)
result = ''
for i in output :
    a = np.argmax(i)
    result += alphabet_dict[str(a)]     # converts encoded integer to corresponding string
    print(a)
print(" Plate number is : ", result)
