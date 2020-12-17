from PIL import Image
import glob
import cv2
import os
image_list = []
name = 'test'
path = 'images'
print(path)
directory = os.path.join(path, name)
for filename in glob.glob(directory + '/*.jpg'): 
    im=Image.open(filename)
    image_list.append(im)
path = ("C:/Users/ledao/Desktop/face_reg/test")

for image in image_list:
    print(image)
image_list[0].save("out3.gif", save_all=True, append_images=image_list[1:], duration=200, loop=0)
            

