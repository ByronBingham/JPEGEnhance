'''
This script creates a dataset from a flat directory full of png images for the JPG to PNG NN.

The script looks in the specified directory. For each PNG file found, a folder is created in the specified folder for
the finished dataset. In that directory, the original PNG is stored and 5 JPG's of quality 100% to %20 are created as
training data corresponding to the original PNG.

Author: Byron Bingham
'''

import os
import shutil as sh
import PIL as pil

from PIL import Image

# get directories from the user

print("Enter the directory to get images from: \n")
imgsDir = input()

print("Enter the directory to create the dataset in: \n")
dsDir = input()

# create the directory for the dataset if it does not already exist

try:
    os.makedirs(dsDir,mode=777,exist_ok=True)
except Exception as e:
    print("Could not create directory for dataset: " + e)
else:
    print("Dataset directory successfully created")

# get list of images to convert to dataset

print(os.listdir(imgsDir))


originalFiles = [f for f in os.listdir(imgsDir)
                 if os.path.isfile(os.path.join(imgsDir,f))]
# and os.path.splitext(f)[1] is ".png"]

print(originalFiles)

# create dataset

for img in originalFiles:

    # create directory for image
    iDir =os.path.join(dsDir,os.path.splitext(img)[0])
    os.makedirs(iDir,mode=777,exist_ok=True)

    # copy PNG to directory
    sh.copy(os.path.join(imgsDir, img),os.path.join(iDir,img))

    # create JPG's
    try:
        im = Image.open(os.path.join(iDir, img))
    except Exception as e:
        print(e)
    else:
        im = im.convert("RGB")

        im.save(os.path.join(dsDir,os.path.splitext(img)[0],os.path.splitext(img)[0] + "_(1).jpg"), format="JPEG", quality=100)
        im.save(os.path.join(dsDir, os.path.splitext(img)[0], os.path.splitext(img)[0] + "_(2).jpg"), format="JPEG", quality=80)
        im.save(os.path.join(dsDir, os.path.splitext(img)[0], os.path.splitext(img)[0] + "_(3).jpg"), format="JPEG", quality=60)
        im.save(os.path.join(dsDir, os.path.splitext(img)[0], os.path.splitext(img)[0] + "_(4).jpg"), format="JPEG", quality=40)
        im.save(os.path.join(dsDir, os.path.splitext(img)[0], os.path.splitext(img)[0] + "_(5).jpg"), format="JPEG", quality=20)

