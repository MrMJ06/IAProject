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

    # Sort the image names by the number in the name
    image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

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
    keyframes = write_keyframes(name_img, name_histr, centroids, classified_images, outputpath)

    # With the keyframes we reconstruct the summary video and write it in outpath
    write_video(keyframes, inputpath, outputpath, s, name_histr, name_img, image_names, H, classified_images)


""" ----------------------- Calc_ histr -------------------------------- """


def calc_histr(inputpath, T, image_names, H):

    # First of all initialize the name - histogram and name - image dictionaries
    name_histogram = dict()
    name_image = dict()

    color = ('v', 'h', 's')  # Initiate the channels in the image

    # Iterate in every image name index with a skip of T frames
    for i in range(0, len(image_names), T):

        histr = dict()  # Initiate the calculated histogram

        input_path = os.path.join(inputpath, image_names[i])  # Concatenate the inputpath and the name of the image

        img = cv2.imread(input_path)  # Read the array image in RGB
        name_image[image_names[i]] = img  # Saves the image in the dictionary
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for j, col in enumerate(color):  # For every color chanel calculate the histgram

            # We use the map function to correct the histogram format and saves it the the histogram color channel
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([hsv], [j], None, [H], [0, H])))

        # When the 3 color channels are calculated in the histogram we saves it in the image hisstogram dictionary
        name_histogram[image_names[i]] = histr

    return [name_histogram, name_image]


""" ------------------------- K-Means --------------------------- """


def k_means(K, name_histr, iterations, n, centroids=None):

    last_classification = None  # The centroids of the last iteration
    classified_images = dict()
    color = ('v', 'h', 's')

    for it in range(0, iterations):  # The iterations will be limited

        print('Iteration number: ' + str(it))

        if centroids is None:  # If the initial centroids are not passed as parameter calculate k random centroids
            centroids = calc_random_centroids(K, n, name_histr)  # Return the random centroids

        classified_images = classify_images(centroids, name_histr)  # Classify the images

        # Update the centroids with the classified images
        centroids = update_centroids(classified_images, name_histr, centroids)

        if last_classification is None:  # If is the first iteration set the last centroids to this centroids
            last_classification = classified_images
        else:
            # If the centroids has not changed the stop
            if list(last_classification.values()) == list(classified_images.values()):
                break
            else:  # In other case continue
                last_classification = classified_images

    return [centroids, classified_images]


""" ---------------------------- Random centroids ----------------------------- """

# Return k random centroids making the mean of three random frames


def calc_random_centroids(K, n, name_histr):

    centroids = dict()
    color = ('v', 'h', 's')  # Initialize the color channels

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

        for col in color:   # Show the histogram
            plt.plot(histr[col])
        plt.show()

        centroids[i] = histr

    return centroids


""" ---------------------------- Classify images ----------------------------- """

# Return a dict with the K index of the centroid and the name of the image


def classify_images(centroids, name_histr):

    classified_images = {k: [] for k, k_histr in centroids.items()}  # Initialize the classified K - image dictionary
    color = ('v', 'h', 's')  # Initialize the color channels

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
            new_histr = {key: [values / len(classified_images[k]) for values in name_histr[name][key]] for key in
                         name_histr[name]}
            if histr is None:
                histr = new_histr
            else:
                histr = {key_h: [sum(x) for x in zip(*[histr[key_h], new_histr[key_h]])] for key_h in histr}

        if len(classified_images[k]) > 0:  # If the centroid has images inside then is updated
            centroids[k] = histr

    print('New centroids : ' + str(centroids))

    return centroids


def write_keyframes(name_img, name_histr, centroids, classified_images, output):

    if not os.path.isdir(output):  # If the directory not exists, creates the directory
        os.makedirs(output)
    else:
        image_names = os.listdir(output)
        for name in image_names:  # Remove the content of the directory
            os.remove(output+'/'+name)

    color = ('v', 'h', 's')
    keyframes = dict()

    for i, k_histr in centroids.items():  # First iterate over all the centroids

        min_value = float('inf')

        for name in classified_images[i]:  # Now we calculate the nearest image

            value_accumulated = 0

            for j, col in enumerate(color):  # Calculate the euclidean distance
                value_accumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(name_histr[name][col], k_histr[col])], 2)))

            if value_accumulated < min_value:  # Update the minimum value and the keyframes if is closer
                min_value = value_accumulated
                keyframes[i] = name

    print('Writing images: ' + str(keyframes))
    for i, name in keyframes.items():  # Write the images in the output
        cv2.imwrite(output + '/' + name, name_img[name])

    return keyframes


def write_video(keyframes, input, output, S, name_histr, name_img, names, H, classified_images):

    img = list(name_img.values())[0]
    heigth, width, channels = img.shape  # Obtain the dimensions of the frames to the video

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Generate the fourcc code nedded to write the video

    # Initialize the video writer
    video = cv2.VideoWriter(output + '/result.avi', fourcc, 30.0, (width, heigth))

    # Sort the keyframes names by the number in his name
    keyframes_names = list(set(keyframes.values()))
    keyframes_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for name in keyframes_names:  # First iterate over the names

        keyframe_k = None

        for k, keyframe in keyframes.items():  # Now obtain his classification (K)
            if name == keyframe:
                keyframe_k = k

        if keyframe_k is not None:  # We calculate an extra value in base of the images of the centroid
            extra = len(classified_images[keyframe_k])
        else:
            extra = 0

        extra = int(5*np.log2(extra)) + S  # Extra frames added in base of number of classified frames TODO: DOCUMENT

        print(extra)
        index = names.index(str(name))  # Obtain the index inside the list for the name
        up_index = index + extra  # The top limit
        down_index = index - extra  # The bottom limit

        # Checks that we not surpass the limits
        if up_index > len(names):
            up_index = len(names)
        if down_index < 0:
            down_index = 0

        # For every image apply KNN algorithm to the keyframes
        for image in names[down_index:up_index]:
            min_k = calc_knn(keyframes, name_histr, image, H)
            if name == keyframes[min_k]:
                print('writing:' + image)
                video.write(cv2.imread(input + '/' + image))

    # Needed functions after call videoWriter
    video.release()
    cv2.destroyAllWindows()

    print('\n\nOK your video is available')


def calc_knn(keyframes, name_histr, image_name, H):

    image_histr = dict()
    color = ('v', 'h', 's')
    image_path = os.path.join(inputpath, image_name)
    img = cv2.imread(image_path)  # Read the image to classify
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Make the conversion of BGR to HSV color base

    for j, col in enumerate(color):  # Calculate the histogram
        image_histr[col] = list(map(lambda x: x[0], cv2.calcHist([hsv], [j], None, [H], [0, H])))

    min_value = float('inf')
    min_k = -1

    for k, name in keyframes.items():  # Calculates the nearest keyframe and return his classification (K)
        value = 0
        for j, col in enumerate(color):  # Calculate the euclidean distance
            value += sum(
                np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(name_histr[name][col], image_histr[col])], 2)))

        if value < min_value:
            min_value = value
            min_k = k

    return min_k


if __name__ == '__main__':
    inputpath = "C:/Users/Manue/Documents/IA/ProyectoIA/test"
    outputpath = "C:/Users/Manue/Documents/IA/ProyectoIA/resultsNuminous"
    main(inputpath, 10, 5, 256, 1000, 3, 30, outputpath)