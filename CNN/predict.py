from tensorflow.keras.models import  load_model
import numpy as np
import data_process
import cv2 as cv
import os
import  datetime
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# for converting encoded integer to string
alphabet_dict = { "0" : '0' , '1':'1', '2':'2' , '3':'3','4' :'4' ,'5' :'5' ,'6' :'6','7':'7' ,'8':'8' ,'9':'9' ,'10':'BE' ,
                '11':'CHE' ,'12':'DAL','13':'EIN' ,'14':'FE' ,'15':'GAF' ,'16':'GHAF' ,'17':'GHEIN' ,'18':'HA' ,'19':'HE' ,'20':'JIM','21' : 'KAF' ,
                  '22':'KHE','23' : 'LAM',
                  '24':'MIM' ,'25':'NON' ,'26':'PE' ,'27':'RE' ,'28':'SAD' ,'29':'SE' ,'30':'SHIN' ,'31':'SIN' ,'32':'TA' ,'33':'TE' ,
                    '34':'VAV' ,'35':'YE' ,'36':'ZA' ,'37':'ZAD' ,'38':'ZAL' ,'39':'ZE' ,'40':'ZHE'   }


# loading model
model = load_model('cnn_classifier2.h5')


# reads image and  gets characters out of it

path = 'roya_bold/10BE21865_base211.png'
img = data_process.img_init(path)
# print('segmenting...')
characters = data_process.plate_segmenter(img,path)
# print(len(characters))
characters = characters.reshape(characters.shape[0],30,30,1)

output = model.predict(characters)              # predicting in first time is slower so I added this

# predicting and getting output string
a = input('ready to predict. press any key...')
start_time = datetime.datetime.now()    # to estimate predicting latency
output = model.predict(characters)
result = ''
for i in output :
    a = np.argmax(i)
    result += alphabet_dict[str(a)]         # converts encoded integer to corresponding string
print("*** Plate number is : ", result,"***")

# estimate predicting latency
end_time = datetime.datetime.now()
timedelta = end_time - start_time
timedelta = str(timedelta).split(":")[2]
print(f'predicted in: {timedelta} seconds')
