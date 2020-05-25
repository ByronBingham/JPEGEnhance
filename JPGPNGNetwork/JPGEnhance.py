# import os
import tensorflow as tf
from modules.Dataset import DataSet, WINDOW_SIZE

import numpy as np
import datetime as dt

imagesToTrain = 100000
evalFrequency = 25


class JPGEnhance:

    def __init__(self):
        self.model = None
        self.checkpointPath = None
        self.dataSet = DataSet()

    def main(self):

        tf.compat.v1.enable_control_flow_v2()
        print("eager or not: " + str(tf.executing_eagerly()))

        # create network
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print(tf.config.experimental.list_physical_devices('GPU'))

        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception as e:
            print(e)

        inputs = tf.keras.Input(shape=(WINDOW_SIZE, WINDOW_SIZE, 3))

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=2)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=1)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), strides=1)(conv2)

        flatConv = tf.keras.layers.Flatten()(conv3)
        flatInput = tf.keras.layers.Flatten()(inputs)

        inputAndConv = tf.keras.layers.Concatenate()([flatConv, flatInput])

        dense1 = tf.keras.layers.Dense(units=WINDOW_SIZE * WINDOW_SIZE * 3,
                                       input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3),
                                       use_bias=True)(inputAndConv)

        dense2 = tf.keras.layers.Dense(units=WINDOW_SIZE * WINDOW_SIZE * 3,
                                       input_shape=(WINDOW_SIZE, WINDOW_SIZE, 3),
                                       use_bias=True)(dense1)

        out = tf.keras.layers.Reshape(target_shape=(WINDOW_SIZE, WINDOW_SIZE, 3))(dense2)

        self.model = tf.keras.Model(inputs=inputs, outputs=out)

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        self.model.summary()

        tf.keras.utils.plot_model(self.model, to_file='model.png')

        # load model checkpoint if exists
        self.checkpointPath = "./JPGEnhanceCheckpoint_" + str(WINDOW_SIZE)

        try:
            self.model.load_weights(self.checkpointPath)
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            print(str(e))

        # load data set
        self.dataSet.startFillQueue()
        print("Finished initializing dataset")

        # start training
        i = 0
        while True:  # i < imagesToTrain:
            try:
                self.trainStep()
            except Exception as e:
                print("Training step failed. Skipping step")
                print(str(e) + "\n")
                continue

            accuracy = self.model.metrics
            print("Training step " + str(i) + " finished\n")

            if i % evalFrequency == 0:
                print("Beginning eval step")
                try:
                    self.evalStep()
                except Exception as e:
                    print("Evaluation step failed. Skipping step")
                    print(str(e) + "\n")
                    continue

            i += 1

        DataSet.destroy()

    def clipData(self, data):
        for i in range(0, len(data)):

            for w in range(0, WINDOW_SIZE):
                for h in range(0, WINDOW_SIZE):
                    for p in range(0, 3):
                        if data[i][w][h][p] > 255.:
                            data[i][w][h][p] = 255.
                        if data[i][w][h][p] < 0.:
                            data[i][w][h][p] = 0.

        return data

    def getEvalCounter(self):
        # get number of the evaluation
        number = 0

        evalCounterFilename = "./evalCounter_" + str(WINDOW_SIZE) + ".cfg"

        # open file
        file = None
        try:
            file = open(evalCounterFilename, "r")
        except Exception as e:
            print("Could not open counter file. Initializing new file...")
            print(str(e))

            try:
                file = open(evalCounterFilename, "x")
                file.close()
                file = open(evalCounterFilename, "w")
                file.write("0")
                file.close()
                file = open(evalCounterFilename, "r")
            except Exception as e:
                print("Probably a filesystem error")
                print(e)

        # read from file
        value = int(file.readline())
        file.close()

        # update count
        try:
            file = open(evalCounterFilename, "w")
            file.write(str(value + 1))
            file.close()
        except Exception as e:
            print("Could not update evaluation counter")
            print(str(e))

        return value

    def evalStep(self):
        trainData, targetData, originalSize = self.dataSet.getBatch()

        trainData = np.asarray(trainData)
        targetData = np.asarray(targetData)

        inputImg = trainData[len(trainData) - 1]
        prediction = self.model.predict(inputImg)

        # save output
        number = self.getEvalCounter()
        prediction = self.clipData(prediction)
        prediction = np.array(prediction).astype(dtype=np.uint8)
        DataSet.saveResults(targetData, inputImg, prediction, originalSize, number)
        print("Evaluation step finished\n")

    def trainStep(self):
        trainData, targetData, originalSize = self.dataSet.getBatch()
        counter = 0

        trainData = np.asarray(trainData)
        targetData = np.asarray(targetData)

        for i in range(len(trainData)):
            time1 = dt.datetime.now()
            with tf.device('/GPU:0'):
                self.model.fit(trainData[i], targetData, verbose=2, epochs=1, batch_size=4)
            print("Time elapsed training: " + str(dt.datetime.now() - time1))

        self.model.save_weights(self.checkpointPath)
        print("Done with training step")


if __name__ == '__main__':
    enhance = JPGEnhance()
    enhance.main()
