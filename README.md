# writer-identification-

This project focuses on handwriting identification, where the goal is to classify handwritten text to identify the writer. The system uses a dataset of scanned handwritten documents from various writers. The challenge lies in the variability of handwriting styles, the large size of the images, and the need to effectively process and classify this data.

The project involved preprocessing the scanned documents, including noise removal, line segmentation, and character extraction. A deep learning model based on ResNet101 was trained to classify the handwriting by learning distinctive features from the images. The project addresses unique challenges, such as data augmentation due to limited samples from each writer and dealing with large, high-resolution images during the training process.

The model's performance was evaluated using metrics like accuracy, loss, and confusion matrices to interpret its ability to distinguish between different handwriting styles.

Technologies Used:

Image Processing: OpenCV, NumPy, and Scikit-image for segmentation, noise removal, and contour detection.
Deep Learning: TensorFlow, Keras, and the pre-trained ResNet101 architecture for feature extraction and classification.
Data Augmentation: Keras ImageDataGenerator to increase the variability of training samples.
Model Evaluation: Confusion matrices, accuracy, and loss metrics to assess the model's performance on validation and test sets.


