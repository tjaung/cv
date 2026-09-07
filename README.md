# Features and Model

I decided to use Sobel gradients with magnitude and direction as well as LAB values. These features alone seem like they can capture most of the information that I need for anomaly detection. I tried HSV and local binary patterns. HSV did not have as large of a differentiation across classes and local binary patterns seemed to show less information than Sobel features.


The first baseline model I have is a PCA embedding. Its pretty simple. It uses the training data (with some withheld for validation) and gets the feature vectors. It then uses the validation set to get some margin of error. Using some error threshold of percentile, I can calculate squared Euclidean distance and Mahalanois distance to find if a given images featues are within the acceptable range.

I also made a one class SVM for anomaly detection. I use grid search to train a bunch of models of a set of parameters.

It turns out that all of the models label good data as good, and all rust images as bad. They all fail on scratches, which is what I feared since the beginning. Looking at the PCA components, most of the variance is described by magnitude. This is great for rust, but it does not pick up enough detail lighter scratches and scratch patches. The scratches that do get flagged as bad are deeper scratches where the metal can be seen. This makes sense since magnitude will pick up this larger change in brightness. Lighter scratches are more blue, similar to the plate paint, and scratch patches are more brown. I need some stronger feature to pick up scrathces, otherwise this will never work.

I added frangi filters and that seemed to help. I only got one model to work well, an SVM with a linear kernel and ν = 0.01.

I think Im going to try out using a classifier now instead of an anomaly detector. I want to try a classifier for each class and an object detection model for finding scratches and rust.

Classifiers work amazingly. SVMs seemed to perform the best with my pick being a linear kernel SVM with variance_target = 0.95 and C=0.1. 

I also want to use object detection. I will need to split the data again like with classifiers probably. I think I will need to add to the pipeline again, probably doing more thresholding and segmenting to extract features of rust and scratches. I will try the embedded approach first as a baseline, then try some detection models, maybe a CNN. 