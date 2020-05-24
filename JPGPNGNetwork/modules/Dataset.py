import PIL as pil
import os
from random import randrange
from PIL import Image
from os.path import join
import numpy as np
import time
import multiprocessing

WINDOW_SIZE = 32
DATASET_BASE_DIR = "C:\\ImgDataset"


class DataSet:
    processes = []

    NUMBER_OF_CPU_THREADS = 20
    SIZE_OF_QUEUE = 15

    def __init__(self):
        self.datasetDir = [f for f in os.listdir(DATASET_BASE_DIR)
                           if os.path.isdir(os.path.join(DATASET_BASE_DIR, f))]
        self.datasetSize = len(self.datasetDir)
        self.batchQueue = multiprocessing.Queue()
        self.currentlyProcessing = multiprocessing.Value('i', 0)

    def getBatch(self):
        if not self.batchQueue.empty():
            batch = self.batchQueue.get()
            print("Batch Available\n"
                  "Batch size: " + str(len(batch[1])) + "\n"
                  "Size of queue: " + str(self.batchQueue.qsize() - 1))
            return batch
        else:
            time.sleep(1)
            return self.getBatch()

    @classmethod
    def destroy(cls):
        print("Waiting for threads to end...")
        for p in cls.processes:
            p.join()

    def startFillQueue(self):
        for x in range(0, DataSet.NUMBER_OF_CPU_THREADS):
            self.processes.append(
                multiprocessing.Process(target=DataSetProcessor.asyncUpdateDataset,
                                        args=(self.batchQueue, self, self.currentlyProcessing)))

        i = 0
        for p in self.processes:
            print("Starting process" + str(i))
            p.start()
            i += 1

    @classmethod
    def saveResults(cls, target, inputImage, predictions, originalShape, iterarion):
        if len(target) != len(predictions):
            raise IndexError("Prediction's dimensions do not match the Target")
        if len(originalShape) != 2:
            raise IndexError("Invalid original shape")

        width, height = originalShape

        tPixels = cls.reassemble(target, width, height)
        iPixels = cls.reassemble(inputImage, width, height)
        pPixels = cls.reassemble(predictions, width, height)

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

    @classmethod
    def reassemble(cls, pixels, width, height):

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


class DataSetProcessor:
    queue = None
    dataSet = None
    currentlyProcessing = None

    @classmethod
    def asyncUpdateDataset(cls, queue, dataSet, currentlyProcessing):
        cls.queue = queue
        cls.dataSet = dataSet
        cls.currentlyProcessing = currentlyProcessing

        if queue.qsize() + currentlyProcessing.value < DataSet.SIZE_OF_QUEUE:
            currentlyProcessing.value += 1

            print("Creating batch async")
            cls.makeNextBatch()
            print("Thread finished creating batch")

            currentlyProcessing.value -= 1
        else:
            time.sleep(1)

    @classmethod
    def getImageSet(cls, index):
        dir = os.path.join(DATASET_BASE_DIR, cls.dataSet.datasetDir[index])
        images = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(('.png', '.jpg'))]
        return images

    @classmethod
    def getTargetImage(cls, images):
        target = [f for f in images if f.endswith('.png')]
        try:
            return target[0]
        except Exception as e:
            print("No target image found for set\n")
            print(e)

        return None

    @classmethod
    def getTrainingImages(cls, images):
        training = [f for f in images if f.endswith('.jpg')]
        try:
            return training
        except Exception as e:
            print("No training images found for set\n")
            print(e)

        return None

    @classmethod
    def partitionImage(cls, image: pil.Image):
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

    @classmethod
    def makeNextBatch(cls):
        index = randrange(0, cls.dataSet.datasetSize)
        images = cls.getImageSet(index)
        target = Image.open(join(DATASET_BASE_DIR, cls.dataSet.datasetDir[index], cls.getTargetImage(images)))
        training = [Image.open(join(DATASET_BASE_DIR, cls.dataSet.datasetDir[index], f)) for f in
                    cls.getTrainingImages(images)]

        trainingData = []
        for dat in training:
            trainingData.append(cls.partitionImage(dat))
        labeledData = cls.partitionImage(target)

        cls.queue.put((trainingData, labeledData, target.size))
