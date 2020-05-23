import PIL as pil
import os
from random import randrange
from PIL import Image
from os.path import join
import numpy as np

size = input("Enter window size for the network: \n")
WINDOW_SIZE = int(size)
print("Window size: " + str(WINDOW_SIZE))


class DataSet:

    def __init__(self):
        self.datasetBaseDir = "C:\\ImgDataset"  # input("Input base directory for the dataset:\n")
        if not os.path.exists(self.datasetBaseDir):
            print("Invalid path\n")
            self.datasetBaseDir = input("Input base directory for the dataset:\n")
        self.baseDir = self.datasetBaseDir
        self.datasetDir = [f for f in os.listdir(self.datasetBaseDir)
                           if os.path.isdir(os.path.join(self.datasetBaseDir, f))]
        self.datasetSize = len(self.datasetDir)

    def getImageSet(self, index):
        dir = os.path.join(self.baseDir, self.datasetDir[index])
        images = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(('.png', '.jpg'))]
        return images

    def getTargetImage(self, images):
        target = [f for f in images if f.endswith('.png')]
        try:
            return target[0]
        except Exception as e:
            print("No target image found for set\n")
            print(e)

        return None

    def getTrainingImages(self, images):
        training = [f for f in images if f.endswith('.jpg')]
        try:
            return training
        except Exception as e:
            print("No training images found for set\n")
            print(e)

        return None

    def partitionImage(self, image: pil.Image):
        pixels = np.asarray(image)
        pixels = pixels.astype('uint8')

        width, height = image.size
        partW = int(width / WINDOW_SIZE)
        if (width % WINDOW_SIZE) > 0:
            partW += 1
        partH = int(height / WINDOW_SIZE)
        if height % WINDOW_SIZE > 0:
            partH += 1

        batches = []

        for w in range(partW):
            for h in range(partH):
                column = []
                for y in range(0, WINDOW_SIZE):

                    row = []
                    for x in range(0, WINDOW_SIZE):
                        try:
                            pixel = pixels[h * WINDOW_SIZE + x][w * WINDOW_SIZE + y][0:3]
                            row.append(pixel)
                        except Exception as e:
                            # print("No pixel at coordinate (" + str(w * WINDOW_SIZE + x) + ", "
                            #      + str(h * WINDOW_SIZE + y) + "). Setting pixel to (0,0,0)\n")
                            # print(e)
                            row.append((0, 0, 0))
                    column.append(row)
                batches.append(column)

        return batches

    def makeNextBatch(self):
        index = randrange(0, self.datasetSize)
        images = self.getImageSet(index)
        target = Image.open(join(self.baseDir, self.datasetDir[index], self.getTargetImage(images)))
        training = [Image.open(join(self.baseDir, self.datasetDir[index], f)) for f in self.getTrainingImages(images)]

        trainingData = []
        for dat in training:
            trainingData.append(self.partitionImage(dat))
        labeledData = self.partitionImage(target)

        return trainingData, labeledData, target.size

    def reassemble(self, pixels, width, height):

        partW = int(width / WINDOW_SIZE)
        if (width % WINDOW_SIZE) > 0:
            partW += 1
        partH = int(height / WINDOW_SIZE)
        if height % WINDOW_SIZE > 0:
            partH += 1

        image = [[[0] * 3] * partW * WINDOW_SIZE] * partH * WINDOW_SIZE
        image = np.array(image)
        for w in range(0, partW):
            for h in range(0, partH):

                for y in range(0, WINDOW_SIZE):
                    for x in range(0, WINDOW_SIZE):
                        image[h * WINDOW_SIZE + y][w * WINDOW_SIZE + x] = pixels[w * partH + h][x][y]

        return image

    def saveResults(self, target, inputImage, predictions, originalShape, iterarion):
        if len(target) != len(predictions):
            raise IndexError("Prediction's dimensions do not match the Target")
        if len(originalShape) != 2:
            raise IndexError("Invalid original shape")

        width, height = originalShape

        tPixels = self.reassemble(target, width, height)
        iPixels = self.reassemble(inputImage, width, height)
        pPixels = self.reassemble(predictions, width, height)

        tPixels = tPixels.astype('uint8')
        iPixels = iPixels.astype('uint8')
        pPixels = pPixels.astype('uint8')

        tImage = pil.Image.fromarray(tPixels, "RGB")
        iImage = pil.Image.fromarray(iPixels, "RGB")
        pImage = pil.Image.fromarray(pPixels, "RGB")

        tImage = tImage.crop((0, 0, width, height))
        iImage = iImage.crop((0, 0, width, height))
        pImage = pImage.crop((0, 0, width, height))

        # create folder for images
        resDir = "./Results_" + str(WINDOW_SIZE)
        if not os.path.exists(resDir):
            os.mkdir(resDir)

        resDir = "./Results_" + str(WINDOW_SIZE) + "/Eval_" + str(iterarion)
        os.mkdir(resDir)

        # save images to new folder
        tImage.save(resDir + "/original.png", format="PNG")
        iImage.save(resDir + "/compressed.png", format="PNG")
        pImage.save(resDir + "/NNOut.png", format="PNG")
