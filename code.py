import cv2
import os
import random
import math
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.image as mpimg
from skimage.io import imread
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input, ResNet101
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2




# -----------------------PREPROCESSING-----------------------
def load_data(peak_file, image_file):
    """
    Load the data files necessary for image preprocessing.

    :param peak_file: path to the peak data file.
    :param image_file: path to the image data file.
    :return: loaded peak and image data.
    """

    stam = loadmat(peak_file)
    peaks_indices = stam['peaks_indices'].flatten()
    index_of_max_in_peak_indices = stam['index_of_max_in_peak_indices'].flatten()[0]
    SCALE_FACTOR = stam['SCALE_FACTOR'].flatten()[0]
    delta = stam['delta'].flatten()[0]
    top_test_area = stam['top_test_area'].flatten()[0]
    bottom_test_area = stam['bottom_test_area'].flatten()[0]

    s = bottom_test_area - top_test_area

    img = mpimg.imread(image_file)

    lines = [(x * SCALE_FACTOR) for i, x in enumerate(peaks_indices)]

    return top_test_area, bottom_test_area, s, img, lines


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def clean_img(img):
    """
    Clean the given image by removing certain noise/artifacts.

    :param img: input image to be cleaned.
    :return: cleaned image.
    """

    l = find_connected_black_cells(img)

    image_copy = np.copy(img)
    for i in l:
        image_copy[i[0], i[1]] = 255

    return image_copy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def dfs(grid, row, col, visited):
    """
    Depth-first search (DFS) function to traverse connected cells in a grid.

    :param grid: input 2D grid to be traversed.
    :param row: current row index.
    :param col: current column index.
    :param visited: list of already visited cells.
    """

    if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]) or [row, col] in visited or grid[row][col] >= 100:
        return

    visited.append([row, col])

    # Recursively check neighboring cells
    dfs(grid, row + 1, col, visited)  # Down
    dfs(grid, row, col - 1, visited)  # Left
    dfs(grid, row, col + 1, visited)  # Right


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def find_connected_black_cells(grid):
    """
    Find connected black cells in a given grid/image.

    :param grid: input 2D grid or image.
    :return: list of coordinates of connected black cells.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Create a 2D array to keep track of visited cells
    visited = []

    # Perform DFS for each black cell in the first row
    connected_black_cells = []
    for r in range(0, 10):
        for col in range(cols):
            if grid[r][col] <= 100 and [r, col] not in visited:
                dfs(grid, r, col, visited)

    return visited


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def checkEmpty(img):
    """
    Check if an image has minimal content or is mostly empty.

    :param img: input image to be checked.
    :return: boolean indicating if the image is mostly empty.
    """
    nonWhite = np.count_nonzero(img < 255)
    white = np.count_nonzero(img == 255)
    return nonWhite / white < 0.015


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def process_image(top, bottom, s, img, lines):
    """
    Process an image by segmenting it based on given parameters.

    :param top: top boundary.
    :param bottom: bottom boundary.
    :param s: scaling factor.
    :param img: input image.
    :param lines: list of line coordinates in the image.
    :return: segmented train, validation, and test images.
    """
    trainAndValidSet = []
    testSet = []
    train_lines = []
    test_lines = []

    testIndex = 0
    for i in range(0, len(lines) - 2):
        if (top >= lines[i] and top <= lines[i + 1]):
            testIndex = i

    for i in range(0, len(lines) - 2):
        if i != testIndex:
            threshold = int((s - (lines[i + 1] - lines[i])))
            ex = 0
            while (threshold % 3 != 0):
                ex += 1
                threshold -= 1

            t = int(lines[i] - (threshold * 1 / 3))
            b = int(lines[i + 1] + ex + threshold * 2 / 3)

            if b - t != s:
                print("error")

            if t not in range(top, bottom) and b in range(top, bottom):
                stea = b - top
                b -= stea
                t -= stea
            elif t in range(top, bottom) and b not in range(top, bottom):
                stea = bottom - t
                b += stea
                t += stea

            train_image = clean_img(img[t:b, :])
            if not checkEmpty(train_image):
                train_lines.append(train_image)
        else:
            test_lines.append(clean_img(img[top:bottom, :]))

    train_valid = segment_words_v12(train_lines, 140)
    test = segment_words_v12(test_lines, 140)

    siz = math.ceil(len(train_valid) * 0.8)

    test_set = [expand(img) for img in div_words_test(test)][:42]

    train_set = [expand(img) for img in div_words_train_valid(train_valid[:siz], 500)][:500]

    valid_set = [expand(img) for img in div_words_train_valid(train_valid[siz:], 100)][:100]

    return train_set, valid_set, test_set


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def segment_words_v12(lines, min_word_width):
    """
    Segment words from given lines using a minimum word width.

    :param lines: list of image lines.
    :param min_word_width: minimum width to consider a segment as a word.
    :return: list of segmented words.
    """

    words = []
    for line in lines:
        _, binary = cv2.threshold(line, 128, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations for better contour detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # Find contours in the image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize an empty list to store the words

        # Iterate over the contours
        for contour in contours:
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            if w >= min_word_width:
                # Extract the word from the line image and threshold
                word = line[:, x:x + w]
                words.append(word)

    return words


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def div_words_train_valid(words, s):
    """
    Divide words into train and validation datasets.

    :param words: list of word images.
    :param s: desired number of word pairs.
    :return: list of random word pairs.
    """

    used_n = []


    # Shuffle the words array
    random.shuffle(words)
    pairs = []
    random_pairs = []
    for firstIndex, firstWord in enumerate(words):
        if firstIndex < len(words) - 1:
            for secondIndex, secondWord, in enumerate(words[firstIndex + 1:]):
                pairs.append(create_img(firstWord, secondWord))

    for i in range(0, s):
        random_pairs.append(pairs[get_unique_random_number(0, len(pairs), used_n)])
    return random_pairs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_unique_random_number(lower_bound, upper_bound, used_numbers=[]):
    """
    Get a unique random number not in the provided list of used numbers.

    :param lower_bound: minimum possible number.
    :param upper_bound: maximum possible number.
    :param used_numbers: list of already used numbers.
    :return: a unique random number.
    """
    available_numbers = [n for n in range(lower_bound, upper_bound) if n not in used_numbers]

    selected_number = random.choice(available_numbers)
    used_numbers.append(selected_number)

    return selected_number


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def div_words_test(words):
    """
    Create word pairs for testing.

    :param words: list of word images.
    :return: list of word pairs for testing.
    """

    # Shuffle the words array
    random.shuffle(words)
    pairs = []
    for firstIndex, firstWord in enumerate(words):
        for secondIndex, secondWord, in enumerate(words):
            if secondIndex != firstIndex:
                pairs.append(create_img(firstWord, secondWord))

    return pairs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_img(firstWord, secondWord):
    """
    Combine two word images with a space in between.

    :param firstWord: first word image.
    :param secondWord: second word image.
    :return: combined image of the two words.
    """

    newPair = []
    space = np.ones((firstWord.shape[0], 60), dtype=np.uint8) * 255
    newPair.append(firstWord)
    newPair.append(space)
    newPair.append(secondWord)
    sen = np.hstack(tuple(newPair))
    _, word_thresholded = cv2.threshold(sen, 128, 255, cv2.THRESH_BINARY_INV)
    return (255 - word_thresholded)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def expand(resized_img):
    """
    Expand the image dimensions and normalize.

    :param resized_img: input image.
    :return: expanded and normalized image.
    """
    expanded_img = np.expand_dims(resized_img, axis=-1)  # Add channel dimension
    return expanded_img.astype('float32') / 255


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def preprocess_and_save():
    """
    Preprocess and save the images after segmenting them into train, validation, and test datasets.
    """
    # Path to data
    pages_directory = "./imgs"
    lines_directory = "./lines"
    pages_files = os.listdir(pages_directory)
    lines_files = os.listdir(lines_directory)

    # Directory to save preprocessed images
    train_save_directory = "./preprocessed_train"
    test_save_directory = "./preprocessed_test"
    validation_save_directory = "./preprocessed_validation"

    create_dir_if_not_exists(train_save_directory)
    create_dir_if_not_exists(test_save_directory)
    create_dir_if_not_exists(validation_save_directory)

    for page_file, line_file in zip(pages_files, lines_files):
        page_path = os.path.join(pages_directory, page_file)
        line_path = os.path.join(lines_directory, line_file)
        top_test_area, bottom_test_area, s, img, lines = load_data(line_path, page_path)
        train, valid, test = process_image(top_test_area, bottom_test_area, s, img, lines)

        train_class_path = "./preprocessed_train/" + page_file
        test_class_path = "./preprocessed_test/" + page_file
        validation_class_path = "./preprocessed_validation/" + page_file

        create_dir_if_not_exists(train_class_path)

        create_dir_if_not_exists(test_class_path)

        create_dir_if_not_exists(validation_class_path)

        # Saving preprocessed images using TensorFlow
        for idx, image in enumerate(train):
            encoded_image = tf.io.encode_png((image * 255).astype('uint8'))
            save_path = os.path.join(train_class_path, f"train_{page_file}_{idx}.png")
            tf.io.write_file(save_path, encoded_image)

        for idx, image in enumerate(valid):
            encoded_image = tf.io.encode_png((image * 255).astype('uint8'))
            save_path = os.path.join(validation_class_path, f"validation_{page_file}_{idx}.png")
            tf.io.write_file(save_path, encoded_image)

        for idx, image in enumerate(test):
            encoded_image = tf.io.encode_png((image * 255).astype('uint8'))
            save_path = os.path.join(test_class_path, f"test_{page_file}_{idx}.png")
            tf.io.write_file(save_path, encoded_image)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dir_if_not_exists(directory_name):
    """
    Create a directory if it doesn't exist.

    :param directory_name: name of the directory to be created.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Running the preprocessing and saving
preprocess_and_save()



#---------------------MODEL TRAINING------------------------------------------

# Paths for train, test, and validation data directories
train_data_dir = "./preprocessed_train"
validation_data_dir = "./preprocessed_validation"

# Hyperparameters
BATCH_SIZE = 32               # Number of samples processed before the model is updated
EPOCHS = 5                    # Number of complete passes through the training dataset
LEARNING_RATE = 0.0002        # Step size at each iteration while moving towards a minimum of the loss function
NUM_CLASSES = len(os.listdir(train_data_dir))  # Number of output classes/categories

# Initialize the training data augmenter with preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Initialize the validation data augmenter with preprocessing
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



# Get the training data from the directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),     # Resizes the input images to the given size
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # Specifies that we have multiple classes
)

# Get the validation data from the directory
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)



# Load the pre-trained ResNet101 model without the top (fully connected) layers
base_model = ResNet101(weights='imagenet', include_top=False)

# Add custom layers with L2 regularization on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Averaging pooling layer
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)               # Regularization method to prevent overfitting
x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.005))(x)

# Create a final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model using the Adam optimizer
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',   # Suitable for multi-class classification
              metrics=['accuracy'])

# Early stopping to prevent overfitting by stopping training once a monitored metric has stopped improving
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

# Reducing the learning rate once learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

callbacks_list = [early_stop, reduce_lr]

# Train the model using the training data generator
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,  # Data to evaluate the model after every epoch
    callbacks=callbacks_list               # List of callbacks to apply during training
)


#-------------------TEST EVALUATION-------------------------------------
test_data_dir = "./preprocessed_test"

# Initialize the test data augmenter with preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Get the test data from the directory
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle = False
)

# Model Evaluation
steps = len(test_generator)
test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


#-----------------------GRAPH PLOTTING-----------------------------------------
# Plot training history
def plot_training_history(history):
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()



# Plot training history
plot_training_history(history)


#---------------------VISUALIZE FIRST CONVNET FILTER----------------------------------

# Find the first convolutional layer in the base_model
conv_layer = None
for layer in base_model.layers:
    if isinstance(layer, Conv2D):
        conv_layer = layer
        break

# Ensure we found a convolutional layer
if conv_layer:
    # Get the weights of the convolutional layer
    conv_weights = conv_layer.get_weights()[0]

    # Normalize the filter weights between 0 and 1 for visualization
    normalized_weights = (conv_weights - np.min(conv_weights)) / (np.max(conv_weights) - np.min(conv_weights))

    num_filters = normalized_weights.shape[3]

    # Plot the filters
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('First Conv Layer Filters')

    for i in range(4):
        for j in range(4):
            if 4 * i + j < num_filters:
                axs[i, j].imshow(normalized_weights[:, :, 0, 4 * i + j], cmap='gray')
                axs[i, j].axis('off')

    plt.show()
else:
    print("Couldn't find a convolutional layer!")


#---------------------CONFUSION MATRIX-----------------------------------------------

# Get the model's predictions on the test data
predictions = model.predict(test_generator)
# Convert softmax outputs to predicted class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class indices from the test generator
true_classes = test_generator.classes
# Get the class labels (names) from the test generator
class_labels = list(test_generator.class_indices.keys())

# Compute the confusion matrix
confusion_mtx = confusion_matrix(true_classes, predicted_classes)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Set the size of the figure
    plt.figure(figsize=(20, 20))

    # Check if normalization is required
    if normalize:
        # Normalize the confusion matrix values
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title("Normalized confusion matrix")
    else:
        plt.title('Confusion matrix, without normalization')

    # Create an annotation array to ensure we don't label small values in the matrix
    annot_array = np.array([["" if val < 0.5 else f"{val:.2f}" for val in row] for row in cm])

    # Plot the heatmap of the confusion matrix
    sns.heatmap(cm, annot=annot_array, fmt=".2f" if normalize else "d",
                cmap=cmap, square=True, linewidths=.5,
                cbar_kws={"shrink": .8},
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')  # Set the y-axis label
    plt.xlabel('Predicted label')  # Set the x-axis label
    plt.show()  # Display the plot


# Call the function to plot the normalized confusion matrix
plot_confusion_matrix(confusion_mtx, class_labels, normalize=True)


#-----------------------ERROR INTERPRETATION-------------------------
# Initialize an empty dictionary to collect information about classifications
# For each true label, this dictionary will keep track of:
# 1. How many times it was misclassified
# 2. How many times it was correctly classified
# 3. A sub-dictionary of all classes it was predicted as, and the count of those predictions
classifications = {}

# Populate the classifications dictionary
for index in range(len(true_classes)):
    # Get the true label for the current instance
    true_label = class_labels[true_classes[index]]

    # If this true label hasn't been encountered before, initialize its entry
    if true_label not in classifications:
        classifications[true_label] = {
            'misclassified': 0,
            'correctly_classified': 0,
            'predicted_classes': {}
        }

    # Get the class this instance was predicted as
    predicted_label = class_labels[predicted_classes[index]]

    # If the prediction is not equal to the true label, increase the misclassified count
    # Otherwise, increase the correctly classified count
    if predicted_label != true_label:
        classifications[true_label]['misclassified'] += 1
    else:
        classifications[true_label]['correctly_classified'] += 1

    # If this predicted class hasn't been recorded for this true label before, initialize its count
    if predicted_label not in classifications[true_label]['predicted_classes']:
        classifications[true_label]['predicted_classes'][predicted_label] = 0

    # Increment the count for this prediction
    classifications[true_label]['predicted_classes'][predicted_label] += 1

# Iterate over each true label in the classifications dictionary
# and print out details for those labels that were misclassified more times than they were correctly classified
for true_label, data in classifications.items():
    if data['misclassified'] > data['correctly_classified']:
        # For each true label, identify which prediction (correct or incorrect) was most frequent
        most_frequent_predicted_class = max(data['predicted_classes'], key=data['predicted_classes'].get)
        # Print the details
        print(f"True Label: {true_label} - Total Predictions: {data['misclassified'] + data['correctly_classified']}")
        print(f"Correctly Classified: {data['correctly_classified']} - Misclassified: {data['misclassified']}")
        print(
            f"Most Frequent Predicted Class: {most_frequent_predicted_class} (Count: {data['predicted_classes'][most_frequent_predicted_class]})")

