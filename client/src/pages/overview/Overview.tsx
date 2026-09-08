export default function Overview() {
  return <section className="overview">
    <div className="page-intro"><h2>Assignment Overview</h2></div>
    <details className="panel overview-content">
      <summary>Table of contents</summary>
      <ol className="contents-list">
        <li><a href="#dataset"><strong>Dataset viewer <span aria-hidden="true">→</span></strong><span>Browse folders, switch between grid and single images, and overlay ground truth.</span></a></li>
        <li><a href="#preprocessing"><strong>Preprocessing <span aria-hidden="true">→</span></strong><span>Explore training and test metal plates through normalization, thresholding, cleanup, and masking.</span></a></li>
        <li><a href="#features"><strong>Features <span aria-hidden="true">→</span></strong><span>Explore LAB and Sobel heatmaps and compare color distributions across good plates and defect types.</span></a></li>
        <li><a href="#models"><strong>Anomaly detection <span aria-hidden="true">→</span></strong><span>Train normal-only patch PCA, inspect feature CSVs, and evaluate anomaly scores on test plates.</span></a></li>
        <li><a href="#classifiers"><strong>Classifiers →</strong><span>Compare supervised PCA, SVM, and KNN on a mixed 80/20 split.</span></a></li>
        <li><a href="#cnn"><strong>CNN →</strong><span>Train ResNet-18 on segmented plates, compare class metrics, and inspect learned filters and predictions.</span></a></li>
        <li><a href="#comparison"><strong>Comparison and Business Use →</strong><span>Compare saved models on random images and rotations, including prediction time and costly missed defects.</span></a></li>
      </ol>
    </details>
    <article className="panel overview-content overview-summary">
      <h1>READ ME</h1>
      <p>This webpage is my lab notebook for tools to help me create my final models, as well as for documenting my work and thought process for this problem. This lab notebook is not my primary submission either. I made this fullstack tool just to help me visualize the images, steps, and results. As a result, this client and server are AI slop and I don't want to be evaluated on the quality of the architecture and code for this app. With this notebook, I am trying to highlight my thought process and show my ML/AI workflow since that is relevant to the role.</p>
      <p>Each tab in consecutive order shows each step I took to reach my final model and conclusion. I try to explain in each tab why I had to go back to a previous step.</p>
      <h1>TLDR</h1>
      <p>Quick overview of what I did on one page if you dont want to go through every tab. I used the metal plate dataset for this assignment. I made a preprocessing pipeline that segments out the plate, and runs some local features on it (Sobel edge with gradients, magnitude, direction), LAB space histograms, HOG, and Frangi filters. These were to capture the scratch and rust patterns with edges, directions, ridges, and color separations from good, blue paint. These were also chosen to reduce the light reflections off of the plate surfaces. I tested out a luminance normalization filter that moves all lumninance toward the center, where bright luminance is moved down and dark luminance is moved up in an effort to fight the glare.</p>
      <p>I ran three types of models, anomaly detection, classifiers, and CNNs. Anomaly detections used PCA space with simple distance metrics and SVMs. Classifiers used were the same embedded PCA space, SVMs, and KNNs. I chose these because my initial thoughts were that I could solve this problem with some embedded space classification. I tried out CNNs with Resnet-18 architectures more as an experiment to see what features it learns, but they turned out to be strong contenders for a final model. I tried 4 iterations, one simple resnet over the entire image, one that uses cross entropy loss to penalize high confidence, wrong predictions. Then, I tried running it on regions of the plate to try to extract smaller details of scratches and differentiate between local and total rust. I ran a binary model predicting good from defect first as a test, then I ran a 4 class model to detect regions of defects.</p>
      <p>Ultimately, I feel that the CNN with localized regions and 4 classes performed the best, and it showed lots of information like detecting regions of defects and what kinds of defects. However, I ran with a scenario that this model would be deployed in real time. The Patch resnet with 4 classes ran much slower than other models. To get real time speed, I ended up choosing the CNN that evaluates the whole plate standard resnet model. It runs with the best performance with the fastest prediction times.</p>
    </article>
  </section>
}
