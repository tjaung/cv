# Features and Model

I decided to use Sobel gradients with magnitude and direction as well as LAB values. These features alone seem like they can capture most of the information that I need for anomaly detection. I tried HSV and local binary patterns. HSV did not have as large of a differentiation across classes and local binary patterns seemed to show less information than Sobel features.

