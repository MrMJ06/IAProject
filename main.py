import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def main(INPUTPATH, K, T, H, N, OUTPUTPATH):
    INPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/prueba"
    OUTPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    image_names = os.listdir(INPUTPATH)
    print('Image names:')
    print(image_names)
    print('-------------------')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    


def calc_histr(inputpath,T,image_names,H):
    image_histr = []
    for i in range(0, len(image_names), T):
        input_path = os.path.join(inputpath, image_names[i])

        img = cv2.imread(input_path)

        # cv2.imshow('scene', img)
        width, heigth, chanels = img.shape
        print(width)
        print(heigth)
        print(width*heigth)
        print(chanels)

        color = ('b', 'g', 'r')

        histr = dict([])
        for j, col in enumerate(color):
            histr[col] = cv2.calcHist([img], [j], None, [H], [0, H])
            plt.plot(histr[col], color=col)
            plt.xlim([0, H])


        plt.show()
        # print(image_names[i] + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return dict(image_histr)


def k_means(K, image_histr, size, H, iterations, centroids):

    centroids = dict([])
    color = ('b', 'g', 'r')

    if centroids is None:
        for i in range(0, K - 1):
            histr = dict([])
            for j, col in enumerate(color):
                mean = np.random.randint(0, size, size=1)
                stdv = np.random.randint(0, size, size=1)
                histr[col] = cv2.calcHist(np.random.normal(mean, stdv, size), [j], None, [H], [0, H])

            centroids[i] = histr

    classified_images = {k:[] for k, k_histr in centroids}

    for it in range(0, iterations):
        for name, img_histr in image_histr:
            min_histr = []
            min_k = -1
            value_acumulated = 0
            min_value = size
            for k, k_histr in centroids:
                for j, col in enumerate(color):
                    for i in range(0, H):
                        value_acumulated+= np.abs(k_histr[col][i]-image_histr[i])
                    if value_acumulated<min_value:
                        min_value = value_acumulated
                        min_k=k
                        min_histr=k_histr

            classified_images[k].append(name)

        centroids = dict([])
        for k in range(0, K - 1):
            histr = dict([])
            for j, col in enumerate(color):
                for i in range(0, H):
                    mean = 0
                    for name in classified_images[k]:
                        mean += image_histr[name][col][i]/len(classified_images[k])

            centroids[k]=histr

    return centroids


main(None, 2, 1, 256, 100,None)

