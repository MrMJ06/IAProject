import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from threading import Thread


def main(INPUTPATH, K, T, H, N, OUTPUTPATH):
    INPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/prueba"
    OUTPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    image_names = os.listdir(INPUTPATH)
    print('Image names:')
    print(image_names)
    print('-------------------')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    k_means(K, image_histr, int(sum(image_histr['images-0893.jpg']['b'])), H, 100, None)



def calc_histr(inputpath,T,image_names,H):
    image_histr = []
    for i in range(0, len(image_names), T):
        input_path = os.path.join(inputpath, image_names[i])

        img = cv2.imread(input_path)
        # cv2.imshow('scene', img)
        # width, heigth, chanels = img.shape
        color = ('b', 'g', 'r')

        histr = dict()
        for j, col in enumerate(color):
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))
            plt.plot(histr[col], color=col)
            plt.xlim([0, H])

        plt.show()
        # print(str(histr) + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return dict(image_histr)


def k_means(K, image_histr, size, H, iterations, centroids = None):

    color = ('b', 'g', 'r')
    for it in range(0, iterations):
        if centroids is None:
            centroids = dict()
            for i in range(0, K):
                histr = dict()
                for j, col in enumerate(color):

                    plt.plot(histr[col])
                plt.show()

                centroids[i] = histr

        print(image_histr)
        classified_images = {k:[] for k, k_histr in centroids.items()}
        print(it)
        for name, img_histr in image_histr.items():
            min_histr = []
            min_k = -1
            value_acumulated = 0
            min_value = float('inf')
            for k, k_histr in centroids.items():
                for j, col in enumerate(color):
                    value_acumulated += sum(np.sqrt(np.power(k_histr[col]-img_histr[col], 2)))

                print(str(value_acumulated)+' K:' + str(k))
                if value_acumulated < min_value:
                    min_value = value_acumulated
                    min_k = k
                    min_histr = k_histr
            classified_images[min_k].append(name)
        print(classified_images)

        centroids = dict()
        for k in range(0, K - 1):
            histr = dict()
            for j, col in enumerate(color):
                for i in range(0, H):
                    mean = 0
                    for name in classified_images[k]:
                        mean += image_histr[name][col][i]/len(classified_images[k])

            centroids[k] = histr

    return centroids


if __name__ == '__main__':
    main(None, 2, 1, 256, 100,None)

