# Imports
import numpy as np
from skimage import img_as_ubyte #convert float to uint8
from skimage.color import rgb2gray
import cv2
import imutils
import time
from time import sleep
from imutils.video import VideoStream
import tensorflow as tf
from tensorflow import keras
from sense_hat import SenseHat

# SenseHat Settings 
sense = SenseHat()
sense.set_rotation(180)
sense.low_light = True

# Load the model 
model= tf.keras.models.load_model('my_model.h5')
print('Model loaded succesfully')

# Start video
vs = VideoStream(0).start()

# Reshape image so that it will fit for our model 
def PreProcess(orig):
    gray = rgb2gray(orig) # converts original to gray image
    gray_u8 = img_as_ubyte(gray)    # converts gray image to uint8
    (thresh, im_bw) = cv2.threshold(gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resized = cv2.resize(bw,(28,28))
    gray_invert = 255 - resized
    
    #Filters image to get rid of noise
    filtered_image = gray_invert
    for row in range(28):
            for col in range(28):
                if gray_invert[row,col] <180:
                    filtered_image[row,col] = 0

    im_final = filtered_image.reshape(1,28,28,1)

    # Predict
    ans = model.predict(im_final)
    print(ans)
    acc = max(ans[0].tolist())

    # Choose digit with highest probability
    ans = ans[0].tolist().index(max(ans[0].tolist()))
    con = acc*100


    print('Predicted digit is:', ans)
    
    sense.show_message(str(ans), text_colour=[255, 0, 0])
    sense.show_message(str(con), scroll_speed = 0.05, text_colour=[0, 0, 255])
    sense.clear()  


# Infinite loop
while True:
    # Resize video stream frames to a max width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # Show frame window
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"): # If `q`, break from the loop
        break
        cv2.destroyAllWindows()
        vs.stop()
     
    elif key == ord("t"): # If 't' key, save image 
        cv2.imwrite("store.jpg", frame)  
        orig = cv2.imread("store.jpg")
        PreProcess(orig)

# Out of loop
vs.stop()
cv2.destroyAllWindows() 