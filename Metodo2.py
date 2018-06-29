import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tkinter.filedialog import askdirectory
import sys
from os import listdir
from os.path import isfile, join
import time
from appJar import gui
import math
import threading

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


def summ_allvideos(inputpath, K, T, H, it, n, s):
    videos = [f for f in listdir(inputpath) if isfile(join(inputpath, f))]
    for x in videos:
        if x.endswith(".mp4") or x.endswith(".mpg") or x.endswith(".avi"):
            print("Summarizing video: " + x)
        summ_Video(inputpath, K, T, H, it, n, s, x)


def summ_Video(inputpath, K, T, H, it, n, s, videoName):
    output_gui("========= Start video summarization : "+ videoName + " =============")
    timestart2 = time.time()
    framesfolder = get_frames_folder(inputpath, videoName)
    print(framesfolder)
    if not os.path.isdir(framesfolder):
        os.makedirs(framesfolder)

    image_names = os.listdir(framesfolder)  # Reads the image names
    if len(image_names) < 3:  # Frames already generated, start algorithm
        video_to_frames(inputpath + "/" + videoName, framesfolder)
        image_names = os.listdir(framesfolder)  # Reload image list after creating it

    # Sort the image names by the number in the name
    image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Readed images:'+str(image_names))
    print('\n')

    # Returns a dict with the name and the histogram of every image
    # and another dict with the name and the image array readed
    name_histr_name_img = calc_histr(framesfolder, T, image_names, H)

    name_histr = name_histr_name_img[0]  # Save the name - histogram dictionary in a variable
    name_img = name_histr_name_img[1]  # Save the name - image array dictionary in a variable

    # Now apply K-Means to the images and return the centroids
    centroids, classified_images = k_means(K, name_histr, it, n)

    # With the centroids we obtain the keyframes and write them in outpath
    keyframes = write_keyframes(name_img, name_histr, centroids, classified_images, get_keyframes_result_folder(inputpath, videoName))

    # With the keyframes we reconstruct the summary video and write it in outpath
    if not os.path.isdir(get_videos_result_folder(inputpath)):
        os.makedirs(get_videos_result_folder(inputpath))
    write_video(keyframes, framesfolder, s, name_histr, name_img, image_names, H, classified_images, get_videos_result_folder(inputpath) + "/summarized-" + videoName.split('.')[0])
    output_gui("========= FINISH : " + videoName + " (in " + str(math.trunc(time.time()-timestart2)) +"s) =============")

""" ----------------------- Calc_ histr -------------------------------- """

def calc_histr(inputpath, T, image_names, H):
    timestart3 = time.time()
    # First of all initialize the name - histogram and name - image dictionaries
    name_histogram = dict()
    name_image = dict()

    color = ('v', 'h', 's')  # Initiate the channels in the image
    progress(0, len(image_names)/T, 'Calculating histograms')
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
        progress(len(name_histogram), len(image_names) / T, 'Calculating histograms')
    output_gui("Calculate histograms finished in: " + str(math.trunc(time.time() - timestart3)) + "s")
    return [name_histogram, name_image]


""" ------------------------- K-Means --------------------------- """


def k_means(K, name_histr, iterations, n, centroids=None):
    timestart4 = time.time()
    last_classification = None  # The centroids of the last iteration
    classified_images = dict()

    for it in range(0, iterations):  # The iterations will be limited

        progress(it, iterations, 'Classifiying frames')
        #  print('Iteration number: ' + str(it))

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
                progress(iterations, iterations, 'Classifiying frames')
                break
            else:  # In other case continue
                last_classification = classified_images
    output_gui("K means task finished in: " + str(math.trunc(time.time() - timestart4)) + "s")
    return [centroids, classified_images]


""" ---------------------------- Centroids jumping ----------------------------- """
# Return k random centroids jumping between frames

def calc_jumping_centroids(K, n, name_histr):
    timestart5 = time.time()
    centroids = dict()
    color = ('v', 'h', 's')  # Initialize the color channels

    centroids = dict()
    s = math.trunc(len(name_histr) / K)
    for i in range(0, K):
        centroids[i] = list(name_histr.values())[i * s]
    output_gui("Calc jumping centroids finished in: " + str(math.trunc(time.time() - timestart5)) + "s")
    return centroids


""" ---------------------------- Random centroids ----------------------------- """

# Return k random centroids making the mean of three random frames


def calc_random_centroids(K, n, name_histr):
    timestart6 = time.time()
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
    output_gui("Calc random centroids finished in:" + str(math.trunc(time.time() - timestart6)) + "s")
    return centroids


""" ---------------------------- Classify images ----------------------------- """

# Return a dict with the K index of the centroid and the name of the image


def classify_images(centroids, name_histr):
    timestart7 = time.time()
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
    output_gui("Classify images finished in: " + str(math.trunc(time.time() - timestart7)) + "s")
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
    timestart8 = time.time()
    if not os.path.isdir(output):  # If the directory not exists, creates the directory
        os.makedirs(output)
    else:
        image_names = os.listdir(output)
        for name in image_names:  # Remove the content of the directory
            os.remove(output+'/'+name)

    color = ('v', 'h', 's')
    keyframes = dict()

    for i, k_histr in centroids.items():  # First iterate over all the centroids

        progress(i, len(centroids), 'Calculating keyframes')
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
        progress(i, len(keyframes), 'Writing keyframes')
        cv2.imwrite(output + '/' + name, name_img[name])
    output_gui("Write keyframes finished in: " + str(math.trunc(time.time() - timestart8)) + "s")
    return keyframes


def write_video(keyframes, input, S, name_histr, name_img, names, H, classified_images, output):
    timestart9 = time.time()
    img = list(name_img.values())[0]
    heigth, width, channels = img.shape  # Obtain the dimensions of the frames to the video

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Generate the fourcc code nedded to write the video

    # Initialize the video writer
    video = cv2.VideoWriter(output + '.avi', fourcc, 30.0, (width, heigth))

    # Sort the keyframes names by the number in his name
    keyframes_names = list(set(keyframes.values()))
    keyframes_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    i = 0
    for name in keyframes_names:  # First iterate over the names

        i += 1
        progress(i, len(keyframes_names), 'Writing video')
        keyframe_k = None

        for k, keyframe in keyframes.items():  # Now obtain his classification (K)
            if name == keyframe:
                keyframe_k = k

        if keyframe_k is not None:  # We calculate an extra value in base of the images of the centroid
            extra = len(classified_images[keyframe_k])
        else:
            extra = 0

        extra = int(5*np.log2(extra)) + S  # Extra frames added in base of number of classified frames TODO: DOCUMENT

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
            min_k = calc_knn(keyframes, name_histr, image, H, input)
            if name == keyframes[min_k]:
                video.write(cv2.imread(input + '/' + image))
    # Needed functions after call videoWriter
    video.release()
    cv2.destroyAllWindows()
    output_gui("Write video finished in: " + str(math.trunc(time.time() - timestart9)) + "s")
    print('\n\nOK your video is available')


def calc_knn(keyframes, name_histr, image_name, H, inputpath):
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


def get_keyframes_result_folder(input, videoName):
    return input + "/resultFrames/" + videoName.split('.')[0]


def get_videos_result_folder(input):
    return input + "/resultVideos"


def get_frames_folder(input, videoName):
    return get_frames_general_folder(input) + "/" + videoName.split(".")[0]


def get_frames_general_folder(input):
    return input + "/frames"


def create_or_clear_folder(folder):
    if not os.path.isdir(folder):  # If the directory not exists, creates the directory
        os.makedirs(folder)
    else:
        image_names = os.listdir(folder)
        for name in image_names:  # Remove the content of the directory
            os.remove(folder+'/'+name)


def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    timestart10 = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    # Start converting the video
    while cap.isOpened():
        progress(count, video_length, 'Converting video')
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        if ret:
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count + 1), frame)
        count = count + 1
        # If there are no more frames left
        if count > (video_length - 1):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            cv2.destroyAllWindows()
            # Print stats
            progress(count, video_length, 'Video' + input_loc+ ' extracted in '+str((time_end - timestart10)))
            output_gui("Video to frames finished in: " + str(math.trunc(time.time() - timestart10)) + "s")
            break


def progress(count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    updateMeter(percents)
    output = '[%s] %s%s ...%s\r' % (bar, percents, '%', status)
    app.setLabel("Task", status)
    print(output)
    #app.setTitle(output)


# create a GUI variable called app
app = gui("Video Summarization", "900x600")
app.setBg("#f4f3ed")
app.setFont(18)
# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "Video Summarization", 0 , 0, 3 )
app.setLabelBg("title", "black")
app.setLabelFg("title", "white")


def add_file_input(btn):
    filename = askdirectory()
    app.setEntry('Input', filename)


app.addLabel("LabelIN", "Input Path", 1, 0, 1)
app.addEntry('Input', 1, 1, 1)

app.addLabel("LabelK", "K (Centroids)", 2, 0, 1)
app.addEntry('K', 2, 1, 1)

app.addLabel("LabelT", "T (Skipped frames)", 3, 0, 1)
app.addEntry('T', 3, 1, 1)

app.addLabel("LabelH", "H (X histogram size)", 4, 0, 1)
app.addEntry('H', 4, 1, 1)

app.addLabel("LabelI", "Iterations", 5, 0, 1)
app.addEntry('I', 5, 1, 1)

app.addLabel("LabelN", "N", 6, 0, 1)
app.addEntry('N', 6, 1, 1)

app.addLabel("LabelS", "S (Frames for videos)", 7, 0, 1)
app.addEntry('S', 7, 1, 1)



#VALORES POR DEFECTO:
app.setEntry("K", 10)
app.setEntry("T", 5)
app.setEntry("H", 256)
app.setEntry("I", 50)
app.setEntry("N", 3)
app.setEntry("S", 20)

app.addButton('Select folder', add_file_input , 1 , 2,1)
app.addLabel("TaskLabel", "Current Task:", 9, 0, 1)
app.addLabel("Task", "Not working", 9, 1, 1)
app.addMeter("progress", 10, 0, 3)
app.setMeterFill("progress", "green")

app.addLabel("TaskOutput", "Output:", 11, 1, 1)
app.addTextArea("Output", 12, 0, 3)

lastMeter = 0
def updateMeter(percentComplete):
    global lastMeter
    percent = math.trunc(percentComplete)
    if percent != lastMeter:
        app.setMeter("progress", percentComplete)
        lastMeter = percent

# link the buttons to the function called press
t = threading.Thread(target=lambda *args: summ_allvideos(app.getEntry("Input"), int(app.getEntry("K")), int(app.getEntry("T")), int(app.getEntry("H")), int(app.getEntry("I")), int(app.getEntry("N")), int(app.getEntry("S")) ) )
app.addButtons(["Summarize"], t.start, 8,1,1)



def output_gui(text):
    app.setTextArea("Output", "\n" + text, end=True, callFunction=True)


# start the GUI
app.go()
