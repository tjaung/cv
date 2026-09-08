import type { ReactNode } from 'react'

function Writeup({ title, children }: { title: string; children: ReactNode }) {
  return <details className="panel preprocessing-writeup" open>
    <summary>{title}</summary>
    <article className="preprocessing-writeup-content">{children}</article>
  </details>
}

export function AnomalyWriteup() {
  return <Writeup title="Anomaly Detection">
    <p>I started with anomaly detection because the original training set only contains good plates and I thought that I could separate each class far enough if I had the right feature set. I tried PCA reconstruction errors and one class SVM, with thresholds set using the 99th percentile of scores from held out good images.</p>
    <p>I chose to work with regions after I found that entire plate models were performing horribly. A scratch might only take up a small part of the plate so averaging everything together could hide it among the good surface areas. Each patch gets its own anomaly score, which can then be used to flag the plate and highlight where something looks unusual. These regions show differences from the good training patches, not a prediction of a specific defect type.</p>
    <h3>Without luminance normalization</h3>
    <p>I first looked at the results without luminance normalization. I had removed it because the images and histograms did not look different enough to make me think it would matter. With LAB, Sobel, HOG, and Frangi features, most of the models still struggled to catch defects, especially scratches. Across 135 configurations, the median defect recall was 57.7% and the median accuracy was 67.0% on the 97 test images.</p>
    <p>The best observed accuracy came from a linear one class SVM using 32 pixel patches, 99% retained PCA variance, and nu of 0.01. It correctly handled 71 of 97 plates (73.2%), catching 45 of 71 defects and accepting all 26 good plates. It caught every major rust and total rust image, but only 8 of 34 scratched plates. That meant 26 scratched plates were being passed as good, which was not enough for what I wanted.</p>
    <h3>Adding normalization back in</h3>
    <p>I then added luminance normalization back into the color branch and retrained the same 135 configurations. The segmentation mask still comes from the original grayscale image, but it is applied to the color image after moving luminance 10% toward the midpoint. The rest of the full pipeline and the feature set stayed the same.</p>
    <p>The overall improvement was small, where median defect recall went from 57.7% to 59.2%, and median accuracy went from 67.0% to 68.0%. Most configurations still did not work well enough. However, the same linear SVM stood out much more clearly. It now correctly handled 90 of 97 plates (92.8%), catching 64 of 71 defects while still accepting all 26 good plates.</p>
    <div className="model-table-scroll"><table>
      <caption>Same linear one class SVM · 32 pixel patches, 99% PCA variance, nu = 0.01 · 97 test images</caption>
      <thead><tr><th>Result</th><th>Without normalization</th><th>With normalization restored</th></tr></thead>
      <tbody>
        <tr><th>Correct plate decisions</th><td>71 / 97 (73.2%)</td><td>90 / 97 (92.8%)</td></tr>
        <tr><th>Defects flagged</th><td>45 / 71 (63.4%)</td><td>64 / 71 (90.1%)</td></tr>
        <tr><th>Scratches flagged</th><td>8 / 34 (23.5%)</td><td>28 / 34 (82.4%)</td></tr>
        <tr><th>Major rust flagged</th><td>14 / 14 (100%)</td><td>14 / 14 (100%)</td></tr>
        <tr><th>Total rust flagged</th><td>23 / 23 (100%)</td><td>22 / 23 (95.7%)</td></tr>
        <tr><th>Good plates rejected</th><td>0 / 26 (0%)</td><td>0 / 26 (0%)</td></tr>
      </tbody>
    </table></div>
    <p>The biggest change was scratches, which adding normalization back in helped this configuration catch 20 more scratched plates. It did miss one total rust plate that it had caught before, so the improvement was not across every class. It still passed six scratched plates and one rusty plate as good, but this was much closer to the result I was looking for.</p>
    <p>I had only reached this promising result after adding HOG and Frangi, the scratch features I discussed in the Features section. This comparison showed me that their input preprocessing mattered too. Both runs above use those features, so this measures the normalization change rather than proving which feature helped. A small visual difference was not enough to judge its effect on the model.</p>
    <p>For anomaly detection, I would keep normalization for this configuration based on these results. It did not fix most of the other models, and I would still want to test the standout model on new images since I compared so many configurations on this test set. The pipeline selector lets me keep these experiments separate and also try features from raw images.</p>
    <p>Overall, I think that these models I chose are not the right fit for this problem. I am using a pretty simple approach to this anomaly detection, so I think these models are the limitation. There may also be features that can capture scratches better than what I have. There may be more robust anomaly detection models out there, but this is an exploration in classical techniques. The next section will improve on this.</p>
  </Writeup>
}

export function ClassifiersWriteup() {
  return <Writeup title="Classical Classifiers">
    <p>The anomaly detection had only one model that had decent performance, and I knew I could do better. I wanted to see how much it would help to train with examples of the defects too. I think using these classical techniques, but really learning the embedded space for the defects can make up for the weaknesses in the anomaly detection models. I combined the images from the original train and test folders and made a new split of about 80% training and 20% testing, keeping each class represented and identical image content on the same side. With seed 42, this gave me 120 training images and 31 test images. I used five fold cross validation within the training set to compare parameters, keeping the test images out of that process.</p>
    <p>I tried PCA with a nearest centroid classifier, SVM, and KNN to predict good, major rust, scratches, or total rust. They use the full preprocessing pipeline (without normalization) and the same LAB, Sobel, HOG, and Frangi features. I still extract features from regions so that local texture and color patterns are captured, then combine the patch vectors using their mean and standard deviation for an overall image prediction.</p>
    <p>These models generally did much better in my testing. In the earlier run with normalization, the median test accuracy across 42 configurations was 90.3%, and nine configurations correctly classified all 31 test images. After removing normalization and retraining, the median fell to 87.1%, with five configurations getting all 31 right. Some configurations may benefit, but the results do not show an overall improvement for classifiers. The table below shows the best observed test accuracy for each model type, rather than a model chosen only by cross validation.</p>
    <div className="model-table-scroll"><table>
      <caption>Best observed classifier results without luminance normalization · 31 image holdout</caption>
      <thead><tr><th>Model / configuration</th><th>Correct / total</th><th>Scratch recall</th><th>Good plates rejected</th></tr></thead>
      <tbody>
        <tr><th>PCA · 90% variance, Manhattan distance</th><td>28 / 31 (90.3%)</td><td>6 / 7 (85.7%)</td><td>2 / 16</td></tr>
        <tr><th>SVM · 90% variance, linear, C = 0.1</th><td>31 / 31 (100%)</td><td>7 / 7 (100%)</td><td>0 / 16</td></tr>
        <tr><th>KNN · 99% variance, 5 neighbors, uniform weighting</th><td>29 / 31 (93.5%)</td><td>6 / 7 (85.7%)</td><td>1 / 16</td></tr>
      </tbody>
    </table></div>
    <p>Scratches were still an issue for some configurations. Both the PCA and KNN examples above passed one scratched plate as good, while the PCA example rejected two good plates and KNN rejected one. The best observed accuracy fell from 29/31 to 28/31 for PCA and from 30/31 to 29/31 for KNN, while SVM stayed at 31/31. Learning from defect examples looks promising, but this is a different task from anomaly detection. These models learn all four classes and are tested on 31 images, while the anomaly models make good/bad decisions on 97 images. The results are encouraging, but not a direct comparison under identical conditions.</p>
    <p>I also wanted to inspect which regions contributed to a decision. Here, the highlighted patches show how removing a patch changes the score for the predicted class. They are explanations of the whole image decision, not independently classified defect regions. If you go through the results of defect predictions, you will find that a lot of highlighted regions are for the hook its on, which is obviously not a good signal of the plate itself being good or bad. These models while much better at classifying plates into their classes, are harder to explain why they made their predictions, and if I had a case where I would want to flag what part of the plate was bad, these models clearly don't do that well.</p>
  </Writeup>
}

export function CNNWriteup() {
  return <Writeup title="CNN Experiments">
    <p>I initially wanted to try a CNN to see what features it could learn instead of relying on the ones I created manually. I had an earlier run of This was really just a check on my hand crafted features, but it turned out to become a powerful model contender. I started with a pretrained ResNet-18 model from Pytorch, changed the final layer to predict the four plate classes, and fine tuned all layers. It uses the same mixed 80/20 image split as the classifiers. For preprocessing, I only use the segmented color plate, followed by the resize, crop, and normalization expected by the ResNet weights.</p>
    <p>My initial results were actually great (standard model). I had a run (I didn't save it) that had high performance, correctly classifying all good plates, but it classified one major rust as good. Looking at the heatmap, it was missing the rust section entirely.</p>
    <p>My next test used weighted cross entropy, giving defect examples twice the weight of good examples. The baseline already used cross entropy, which penalizes assigning a low probability to the correct class; the change was to make mistakes on defect examples matter more during training. This did not improve those initial test results. It still passed one major rust plate as good.</p>
    <p>I then looked at the fact that this was making one prediction for the entire plate. A small defect might be easier to pick up if the model looked at individual regions, so I tried ResNet on overlapping 64×64 patches with a stride of 32. First, I made it a binary good/bad classifier. Patch labels come from the ground truth defect masks, and images are split before extracting patches so regions from the same image do not end up in both training and testing.</p>
    <p>In the earlier run with luminance normalization, the binary patch version caught all 15 defective test plates, but it also rejected six of the 16 good ones. This was better at keeping defects from being sent out, but it would send a lot of good plates back for another inspection. Its plate decision uses the highest defect probability across patches, so a single false alarm can reject the whole plate.</p>
    <p>Next, I tried four class patch predictions so I could see which type of defect was predicted and where it was. To reduce false alarms on good plates, I also allowed a plate to be called good when no more than two patches were flagged and the average good probability was higher than every defect class. That version correctly classified all 15 defective plates and rejected just one good plate as scratched. It had the same overall accuracy as the original whole plate model, but its mistake was less concerning to me from a business perspective.</p>
    <p>At that point, I went back to preprocessing and tried removing luminance normalization. I first retrained the standard and weighted whole plate models. Both now correctly classified all 31 test images, including the three major rust plates and seven scratched plates. Weighted loss still did not improve the final decisions over the standard version, so I did not see a reason to prefer it based on these results alone.</p>
    <p>I also repeated the patch experiments without luminance normalization. The binary version still caught all 15 defective plates, but its false alarms fell from six good plates to two. The four class patch version correctly classified all 31 plates, with no good plates rejected. These results made me lean toward leaving luminance correction out for CNNs, although one or two images make a large difference on this small test set.</p>
    <div className="model-table-scroll"><table>
      <caption>CNN experiment history · 120 training images and the same 31 image holdout</caption>
      <thead><tr><th>Version</th><th>Training accuracy</th><th>Test correct / total</th><th>Defects flagged / total</th><th>Defect precision</th><th>Good plates rejected / total</th><th>Test macro F1</th></tr></thead>
      <tbody>
        <tr><th>Whole plate · four classes · with luminance normalization</th><td>—</td><td>30 / 31 (96.8%)</td><td>14 / 15 (93.3%)</td><td>100%</td><td>0 / 16 (0%)</td><td>94.2%</td></tr>
        <tr><th>Whole plate · weighted loss · with luminance normalization</th><td>—</td><td>30 / 31 (96.8%)</td><td>14 / 15 (93.3%)</td><td>100%</td><td>0 / 16 (0%)</td><td>94.2%</td></tr>
        <tr><th>Patches · binary · with luminance normalization</th><td>—</td><td>25 / 31 (80.6%)</td><td>15 / 15 (100%)</td><td>71.4%</td><td>6 / 16 (37.5%)</td><td>80.1%</td></tr>
        <tr><th>Patches · four classes · with luminance normalization</th><td>—</td><td>30 / 31 (96.8%)</td><td>15 / 15 (100%)</td><td>93.8%</td><td>1 / 16 (6.3%)</td><td>97.5%</td></tr>
        <tr><th>Whole plate · four classes · without luminance normalization</th><td>100%</td><td>31 / 31 (100%)</td><td>15 / 15 (100%)</td><td>100%</td><td>0 / 16 (0%)</td><td>100%</td></tr>
        <tr><th>Whole plate · weighted loss · without luminance normalization</th><td>100%</td><td>31 / 31 (100%)</td><td>15 / 15 (100%)</td><td>100%</td><td>0 / 16 (0%)</td><td>100%</td></tr>
        <tr><th>Patches · binary · without luminance normalization</th><td>95.8%</td><td>29 / 31 (93.5%)</td><td>15 / 15 (100%)</td><td>88.2%</td><td>2 / 16 (12.5%)</td><td>93.5%</td></tr>
        <tr><th>Patches · four classes · without luminance normalization</th><td>100%</td><td>31 / 31 (100%)</td><td>15 / 15 (100%)</td><td>100%</td><td>0 / 16 (0%)</td><td>100%</td></tr>
      </tbody>
    </table></div>
    <p>These are plate level metrics. Defect precision is the percentage of flagged plates that really have a defect, combining all defect types. Macro F1 averages over two classes for the binary model and four classes for the others, so those values describe different tasks. The earlier rows retain the previously recorded results; training accuracy is left blank where it is no longer available in the saved runs.</p>
    <p>I also wanted to check whether getting the plate right meant that the model was finding all of the defect regions. It did not. In the latest four class patch run, it correctly labeled 919 of 1,074 annotated scratch patches as scratches. Of the remaining scratch patches, 154 were called good and one was called major rust. Other patches could still flag the plate correctly, which explains how every scratched plate could be caught even when some scratched regions were missed.</p>
    <div className="model-table-scroll"><table>
      <caption>Local patch results · 10,268 test patches from the same 31 plates</caption>
      <thead><tr><th>Patch version</th><th>Patch accuracy</th><th>Patch macro F1</th><th>Local recall</th></tr></thead>
      <tbody>
        <tr><th>Binary · with luminance normalization</th><td>97.7%</td><td>97.2%</td><td>Defects: 2,841 / 3,035 (93.6%)</td></tr>
        <tr><th>Four classes · with luminance normalization</th><td>96.7%</td><td>93.3%</td><td>Scratches: 954 / 1,074 (88.8%)</td></tr>
        <tr><th>Binary · without luminance normalization</th><td>97.7%</td><td>97.2%</td><td>Defects: 2,863 / 3,035 (94.3%)</td></tr>
        <tr><th>Four classes · without luminance normalization</th><td>97.1%</td><td>94.6%</td><td>Scratches: 919 / 1,074 (85.6%)</td></tr>
      </tbody>
    </table></div>
    <p>This also showed a tradeoff where removing luminance correction improved the four class patch model's plate decisions, but its scratch patch recall went down. The patches overlap, so these are not 10,268 independent test images. For now, the standard whole plate model is a simpler option for an overall decision, while the four class patch model also gives me regions to inspect. I would want more runs and new images before deciding which one to rely on. The ResNet input normalization still runs in every version; it is separate from the luminance correction I removed.</p>
  </Writeup>
}
