import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def main(INPUTPATH, K, T, H, it, n, s, OUTPUTPATH):

    INPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/frames"
    OUTPUTPATH = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    image_names = os.listdir(INPUTPATH)
    image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Image names:'+str(image_names))
    print('-------------------\n\n')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    centroids = k_means(K, image_histr[0], H, it, n)
    keyframes = write_keyframes(image_histr[1], image_histr[0], centroids, OUTPUTPATH)
    write_video(keyframes, INPUTPATH, OUTPUTPATH, s, image_histr[1], image_histr[0])



def calc_histr(inputpath, T, image_names, H):
    image_histr = []
    name_image = dict()
    for i in range(0, len(image_names), T):
        input_path = os.path.join(inputpath, image_names[i])

        img = cv2.imread(input_path)
        # cv2.imshow('scene', img)
        # width, heigth, chanels = img.shape
        color = ('b', 'g', 'r')
        name_image[image_names[i]] = img

        histr = dict()
        for j, col in enumerate(color):
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))
        #     plt.plot(histr[col], color=col)
        #     plt.xlim([0, H])
        #
        # plt.show()
        # print(str(histr) + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return [dict(image_histr), name_image]


def k_means(K, image_histr, H, iterations, n, centroids = None):
    color = ('b', 'g', 'r')
    last_centroids = None

    for it in range(0, iterations):
        if centroids is None:
            centroids = dict()
            for i in range(0, K):
                histr = None
                for j in range(0, n):
                    random_hist = np.random.choice(list(image_histr.values()))

                    random_hist = {key: [int(values)/n for values in random_hist[key]] for key in random_hist}
                    if histr is None:
                        histr = random_hist
                    else:
                        histr = {key_h: [sum(x) for x in zip(*[histr[key_h], random_hist[key_h]])] for key_h in histr}

                for col in color:
                    plt.plot(histr[col])
                plt.show()

                centroids[i] = histr

        classified_images = {k: [] for k, k_histr in centroids.items()}
        print('Iteration number: '+str(it))
        for name, img_histr in image_histr.items():
            # min_histr = []
            min_k = -1
            min_value = float('inf')
            for k, k_histr in centroids.items():
                value_acumulated = 0
                for j, col in enumerate(color):
                    value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(img_histr[col], k_histr[col])], 2)))

                if value_acumulated < min_value:
                    min_value = value_acumulated
                    min_k = k
                    # min_histr = k_histr
            classified_images[min_k].append(name)
        print('Classified images: ' + str(classified_images))

        for k in classified_images:
            histr = None
            for name in classified_images[k]:
                if histr is None:
                    histr = {key: [int(values)/len(classified_images[k]) for values in image_histr[name][key]] for key in image_histr[name]}
                else:
                    histr = {key_h: [sum(x)/len(classified_images[k]) for x in zip(*[histr[key_h], image_histr[name][key_h]])] for key_h in histr}
            if len(classified_images[k]) > 0:
                centroids[k] = histr

        print('New centroids : ' + str(centroids))
        if last_centroids is None:
            last_centroids = centroids
        else:
            if last_centroids.values() == centroids.values():
                break
            else:
                last_centroids = centroids
    return centroids


def write_keyframes(images, histograms, centroids, output):

    image_names = os.listdir(output)
    for name in image_names:
        os.remove(output+'/'+name)
    color = ('b', 'g', 'r')
    keyframes = dict()

    for i, k_histr in centroids.items():
        min_value = float('inf')
        for name, histr in histograms.items():
            value_acumulated = 0
            for j, col in enumerate(color):
                value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(histr[col], k_histr[col])], 2)))
            if value_acumulated < min_value:
                min_value = value_acumulated
                keyframes[i] = name
    print('Writed images: ' + str(keyframes))
    for i, name in keyframes.items():
        cv2.imwrite(output+'/'+name, images[name])

    return keyframes


def write_video(keyframes, input, output, amount, images, image_histr):

    color = ('b', 'g', 'r')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img = list(images.values())[0]
    heigth, width, chanels = img.shape
    video = cv2.VideoWriter(output + '/result.avi', fourcc, 30.0, (width, heigth))
    names = list(image_histr.keys())
    names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for k in names:
        index = names.index(str(k))
        upIndex = index + amount
        downIndex = index - amount
        if upIndex > len(names):
            upIndex = len(names)
        if downIndex < 0:
            downIndex = 0
        for image in names[downIndex:upIndex]:
            min_value = float('inf')
            min_keyframe = None
            for i, keyframe in keyframes.items():
                value = 0
                for col in color:
                    value += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(image_histr[keyframe][col], image_histr[image][col])], 2)))
                if value < min_value:
                    min_keyframe = keyframe
                    min_value = value
            if min_keyframe == k:
                print('writing:' + image)
                video.write(cv2.imread(input + '/' + image))
    video.release()
    cv2.destroyAllWindows()
    print('\n\nOK your video is available')


if __name__ == '__main__':
    main(None, 20, 10, 256, 10, 3, 30, None)

