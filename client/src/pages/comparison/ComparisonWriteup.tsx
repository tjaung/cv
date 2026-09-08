export default function ComparisonWriteup() {
  return <details className="panel preprocessing-writeup" open>
    <summary>Comparison and Business Use</summary>
    <div className="preprocessing-writeup-content">
      <p>After testing the models individually, I wanted to compare the ones I saved on the same images and see how long they actually took to make a prediction. A model might have great accuracy, but if it takes too long to inspect each plate, it might not be useful on a manufacturing line. I also randomly rotated some images to see how the models would handle a plate being in a different orientation.</p>

      <p>Ultimately, I think I would choose the standard whole plate ResNet. It has a good balance of performance and speed without needing to run the network over every patch. In my latest comparison of 50 images from the test folder, it classified 49 out of 50 images correctly and took about 45 ms per image. This set had 13 good plates, 37 defective plates, and 27 rotated images. It caught 36 of the 37 defects and did not reject any good plates. The weighted whole plate version had the same overall results and similar speed, so I did not see a reason to choose the weighted loss over the standard model from this comparison.</p>

      <h3>Latest results on 50 images</h3>
      <div className="model-table-scroll"><table>
        <thead><tr><th>Model</th><th>Correct good/bad</th><th>Correct class</th><th>Defects caught</th><th>Good plates rejected</th><th>Mean prediction time</th></tr></thead>
        <tbody>
          <tr><th>One class SVM (e1e89a02)</th><td>48/50 (96%)</td><td>Binary only</td><td>35/37 (94.6%)</td><td>0/13</td><td>445.5 ms</td></tr>
          <tr><th>SVM classifier (3437247f)</th><td>48/50 (96%)</td><td>48/50 (96%)</td><td>35/37 (94.6%)</td><td>0/13</td><td>365.5 ms</td></tr>
          <tr><th>Standard whole plate ResNet</th><td>49/50 (98%)</td><td>49/50 (98%)</td><td>36/37 (97.3%)</td><td>0/13</td><td>45.5 ms</td></tr>
          <tr><th>Weighted whole plate ResNet</th><td>49/50 (98%)</td><td>49/50 (98%)</td><td>36/37 (97.3%)</td><td>0/13</td><td>46.0 ms</td></tr>
        </tbody>
      </table></div>

      <p>The four class patch CNN was also promising, particularly because it could show which regions looked defective, but it was much slower. In an earlier comparison of 10 images, it classified all 10 images correctly but took about 14.3 seconds per image. The standard whole plate ResNet also classified those same 10 images correctly and took about 42 ms per image. Both models predict four classes. The patch model takes longer because it runs over many overlapping regions and then combines the predictions. I like the localized predictions, but that cost seems too high for my first choice of a model to inspect every plate. That earlier run mixed train and test images, and the patch model was not included in the latest run with 50 images, so I cannot say that it performs better on that set.</p>

      <p>From a business perspective, I would care more about a defect passing inspection than a good plate being sent back for another check. The standard ResNet still missed one scratch image after a rotation of 45 degrees, so 98% accuracy does not mean that this problem is solved. My initial choice would be to use the whole plate model for fast inspection and investigate whether a slower patch model or manual review could help with uncertain cases. I would need to test that review rule before assuming it catches the defects that the first model misses.</p>

      <p>One limitation of this comparison is that the classifiers and CNNs were trained using a shuffled split across the original train and test folders. Although new comparison runs sample only from the test folder, some of those images were still used to train those models. I see this as a useful check of speed and sensitivity to rotation, rather than a final estimate of performance on new plates. Before using it on a manufacturing line, I would want a separate set of unseen plates and would focus on scratch recall, missed defects, and whether the prediction time fits the line speed.</p>
    </div>
  </details>
}
