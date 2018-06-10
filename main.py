import os
import cv2
from matplotlib import pyplot as plt



def main(INPUTPATH,K,T,H,OUTPUTPATH):
    INPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/frames"
    OUTPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    image_names = os.listdir(INPUTPATH)
    image_histr = calc_histr(INPUTPATH,T,image_names,H)


def calc_histr(inputpath,T,image_names,H):
    image_histr = []
    for i in range(0, len(image_names), T):
        input_path = os.path.join(inputpath, image_names[i])

        print(input_path)

        img = cv2.imread(input_path)

        cv2.imshow('scene', img)

        color = ('b', 'g', 'r')

        histr = []
        for i, col in enumerate(color):
           histr[col] = cv2.calcHist([img], [i], None, [H], [0, H])

        image_histr.append((image_names[i], histr))

    return dict(image_histr)

def k_means(K,image_histr):
    return None


if __name__== '__main__':
    main()

