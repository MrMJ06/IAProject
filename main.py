import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

""" 
INPUTPATH -> Path for the frames directory
K -> K value for K-Means method
T -> Number of skipped frames reading images
H -> Histogram x-axis size 
it -> Number of max iterations for the K-Means
n -> Number of frames used to make the random centroids for k means
s -> Number of frames after and before choosed for the video restauration
OUTPUTPATH -> Path for the keyframes and video writing
"""


def main(inputpath, K, T, H, it, n, s, outputpath):

    image_names = os.listdir(inputpath)  # Reads the image names
    image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # Sort the image names by the number in the name

    print('Readed images:'+str(image_names))
    print('\n\n')

    # Returns a dict with the name and the histogram of every image
    # and another dict with the name and the image array readed
    name_histr_name_img = calc_histr(inputpath, T, image_names, H)

    name_histr = name_histr_name_img[0]  # Save the name - histogram dictionary in a variable
    name_img = name_histr_name_img[1]  # Save the name - image array dictionary in a variable

    # Now apply K-Means to the images and return the centroids
    centroids, classified_images = k_means(K, name_histr, it, n)

    # With the centroids we obtain the keyframes and write them in outpath
    keyframes = write_keyframes(name_img, name_histr, centroids, outputpath)

    # With the keyframes we reconstruct the summary video and write it in outpath
    write_video(keyframes, inputpath, outputpath, s, name_histr, name_img, image_names, H, centroids)


""" ----------------------- Calc_ histr -------------------------------- """


def calc_histr(inputpath, T, image_names, H):

    # First of all initialize the name - histogram and name - image dictionaries
    name_histogram = dict()
    name_image = dict()

    color = ('b', 'g', 'r')  # Initiate the channels in the image

    # Iterate in every image name index with a skip of T frames
    for i in range(0, len(image_names), T):

        histr = dict()  # Initiate the calculated histogram

        input_path = os.path.join(inputpath, image_names[i])  # Concatenate the inputpath and the name of the image

        img = cv2.imread(input_path)  # Read the array image in RGB
        name_image[image_names[i]] = img  # Saves the image in the dictionary

        for j, col in enumerate(color): # For every color chanel calculate the histgram

            # We use the map function to correct the histogram format and saves it the the histogram color channel
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))

        # When the 3 color channels are calculated in the histogram we saves it in the image hisstogram dictionary
        name_histogram[image_names[i]] = histr

    return [name_histogram, name_image]


""" ------------------------- K-Means --------------------------- """


def k_means(K, name_histr, iterations, n, centroids=None):

    last_centroids = None  # The centroids of the last iteration

    for it in range(0, iterations):  # The iterations will be limited
        # print('Iteration number: ' + str(it))

        if centroids is None:  # If the initial centroids are not passed as parameter calculate k random centroids
            centroids = calc_random_centroids(K, n, name_histr)  # Return the random centroids

        classified_images = classify_images(centroids, name_histr)  # Classify the images

        centroids = update_centroids(classified_images, name_histr, centroids)  # Update the centroids with the classified images

        if last_centroids is None:  # If is the first iteration set the last centroids to this centroids
            last_centroids = centroids
        else:
            if last_centroids.values() == centroids.values():  # If the centroids has not changed the stop
                break
            else:  # In other case continue
                last_centroids = centroids

    return [centroids, classified_images]


""" ---------------------------- Random centroids ----------------------------- """

# Return k random centroids making the mean of three random frames


def calc_random_centroids(K, n, name_histr):

    centroids = dict()
    color = ('b', 'g', 'r')  # Initialize the color channels

    for i in range(0, K):

        histr = None

        for j in range(0, n):

            random_hist = np.random.choice(list(name_histr.values()))  # Select a random histogram from list

            # Divide the histogram by the number of frames used to make the mean
            random_hist = {key: [int(values) / n for values in random_hist[key]] for key in random_hist}

            if histr is None:  # If is the first the histogram is setted
                histr = random_hist
            else:  # In other case the histogram will be the sum of histr and the random histogram generated
                histr = {key_h: [sum(x) for x in zip(*[histr[key_h], random_hist[key_h]])] for key_h in histr}

        for col in color:
            plt.plot(histr[col])
        plt.show()

        centroids[i] = histr

    return centroids


""" ---------------------------- Classify images ----------------------------- """

# Return a dict with the K index of the centroid and the name of the image


def classify_images(centroids, name_histr):

    classified_images = {k: [] for k, k_histr in centroids.items()}  # Initialize the classified K - image dictionary
    color = ('b', 'g', 'r')  # Initialize the color channels

    for name, img_histr in name_histr.items():  # For every image we calculate the nearest K centroid

        min_k = -1
        min_value = float('inf')

        for k, k_histr in centroids.items():  # Now we calculate the nearest K centroid for the image

            value_accumulated = 0

            for j, col in enumerate(color):  # We use the euclidean distance for every color channel
                value_accumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(img_histr[col], k_histr[col])], 2)))

            # If the value accumulated is lower than the minimum value the min k and the min value is updated
            if value_accumulated < min_value:
                min_value = value_accumulated
                min_k = k

        classified_images[min_k].append(name)  # Saves the image in the minimum K centroid
    print('Classified images: ' + str(classified_images))

    return classified_images


""" ---------------------------- Update centroids ----------------------------- """

# Return the new centroids making the mean of the image histograms inside


def update_centroids(classified_images, name_histr, centroids):

    for k in classified_images:

        histr = None
        # Makes the mean of every image histogram
        for name in classified_images[k]:
            if histr is None:
                histr = {key: [int(values) / len(classified_images[k]) for values in name_histr[name][key]] for key in
                         name_histr[name]}
            else:
                histr = {
                key_h: [sum(x) / len(classified_images[k]) for x in zip(*[histr[key_h], name_histr[name][key_h]])] for
                key_h in histr}

        if len(classified_images[k]) > 0:  # If the centroid has images inside then is updated
            centroids[k] = histr

    print('New centroids : ' + str(centroids))

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
                print('')
    print('Writed images: ' + str(keyframes))
    for i, name in keyframes.items():
        cv2.imwrite(output+'/'+name, images[name])

    return keyframes


def write_video(keyframes, input, output, S, name_histr, name_img, names, H, centroids):

    img = list(name_img.values())[0]
    heigth, width, channels = img.shape  # Obtain the dimensions of the frames to the video

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Generate the fourcc code nedded to write the video

    # Initialize the video writer
    video = cv2.VideoWriter(output + '/result.avi', fourcc, 30.0, (width, heigth))

    keyframes_names = list(set(keyframes.values()))
    keyframes_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for name in keyframes_names:
        index = names.index(str(name))  # Obtain the index inside the list for the name
        up_index = index + S  # The top limit
        down_index = index - S  # The bottom limit

        # Checks that we not surpass the limits
        if up_index > len(names):
            up_index = len(names)
        if down_index < 0:
            down_index = 0

        # For every image apply KNN algorithm
        for image in names[down_index:up_index]:
            min_k = calc_knn(keyframes, name_histr, image, H)
            print(image+' K:'+ str(min_k))
            if name == keyframes[min_k]:
                print('writing:' + image)
                video.write(cv2.imread(input + '/' + image))

    video.release()
    cv2.destroyAllWindows()
    print('\n\nOK your video is available')


def calc_knn(keyframes, name_histr, image_name, H):

    image_histr = dict()
    color = ('b', 'g', 'r')
    image_path = os.path.join(inputpath, image_name)
    img = cv2.imread(image_path)

    for j, col in enumerate(color):
        image_histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))

    min_value = float('inf')
    min_k = -1
    for k, name in keyframes.items():
        value = 0
        for j, col in enumerate(color):
            value += sum(
                np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(name_histr[name][col], image_histr[col])], 2)))

        if value < min_value:
            min_value = value
            min_k = k

    return min_k


if __name__ == '__main__':
    inputpath = "C:/Users/Manue/Documents/IA/ProyectoIA/test"
    outputpath = "C:/Users/Manue/Documents/IA/ProyectoIA/results"
    main(inputpath, 15, 60, 256, 100, 3, 30, outputpath)