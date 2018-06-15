import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from threading import Thread


def main(INPUTPATH, K, T, H, it, n, OUTPUTPATH):
    INPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/prueba"
    OUTPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    image_names = os.listdir(INPUTPATH)
    print('Image names:')
    print(image_names)
    print('-------------------')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    k_means(K, image_histr, int(sum(image_histr['images-0893.jpg']['b'])), H, it, n)



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
        #     plt.plot(histr[col], color=col)
        #     plt.xlim([0, H])
        #
        # plt.show()
        # print(str(histr) + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return dict(image_histr)


def k_means(K, image_histr, size, H, iterations, n, centroids = None):

    color = ('b', 'g', 'r')
    for it in range(0, iterations):
        if centroids is None:
            centroids = dict()
            for i in range(0, K):
                histr = None
                random_hist = None
                for j in range(0, n):
                    random_hist = np.random.choice(list(image_histr.values()))

                    random_hist = {key: [int(values)/n for values in random_hist[key]] for key in random_hist}
                    if histr is None:
                        histr = random_hist
                    else:
                        histr = {key_h : [sum(x) for x in zip(*[histr[key_h], random_hist[key_h]])] for key_h in histr}

                for col in color:
                    plt.plot(histr[col])
                plt.show()

                centroids[i] = histr

        classified_images = {k:[] for k, k_histr in centroids.items()}
        print('Iteration number: '+str(it))
        for name, img_histr in image_histr.items():
            min_histr = []
            min_k = -1
            min_value = float('inf')
            for k, k_histr in centroids.items():
                value_acumulated = 0
                for j, col in enumerate(color):
                    value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(img_histr[col], k_histr[col])], 2)))

                if value_acumulated < min_value:
                    min_value = value_acumulated
                    min_k = k
                    min_histr = k_histr
            classified_images[min_k].append(name)
        print('Classified images: ' + str(classified_images))

        centroids = dict()
        for classes in classified_images.values():
            if not classes:
                return centroids

        for k in classified_images:
            histr = None
            for name in classified_images[k]:
                if histr is None:
                    histr = {key: [int(values)/len(classified_images[k]) for values in image_histr[name][key]] for key in image_histr[name]}
                else:
                    histr = {key_h: [sum(x)/len(classified_images[k]) for x in zip(*[histr[key_h], image_histr[name][key_h]])] for key_h in histr}

            centroids[k] = histr
        print('New centroids : '+ str(centroids))
    return centroids


if __name__ == '__main__':
    main(None, 2, 1, 256, 100, 3, None)

