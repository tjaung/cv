import Writeup from '../../components/Writeup'

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
