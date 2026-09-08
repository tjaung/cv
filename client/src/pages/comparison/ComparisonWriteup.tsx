export default function ComparisonWriteup() {
  return <details className="panel preprocessing-writeup" open>
    <summary>Comparison and Business Use</summary>
    <div className="preprocessing-writeup-content">
      <p>By this point, I had tried several ways to solve the assignment, but the main question was still whether a plate was good or bad. The four class predictions and local regions gave me more information, but I wanted to see whether that extra work was worth the time it took. I compared the saved models on the same random images and timed their predictions. I also rotated some images to try a change in orientation, though rotating the whole image is only an approximation of the plate moving in front of a camera.</p>

      <p>Ultimately, I would choose the standard whole plate ResNet for this assignment. It correctly classified all 31 images in the supervised holdout described in the CNN tab, and the timing checks gave me another reason to keep it. It can make one prediction for the plate without running the network over every patch. In the 50 image comparison shown below, it classified 49 out of 50 images correctly and took about 45 ms per image. This set had 13 good plates, 37 defective plates, and 27 rotated images. It caught 36 of the 37 defects and did not reject any good plates. The weighted whole plate version had the same overall results and similar speed, so I did not see a reason to choose the weighted loss over the standard model from this comparison.</p>

      <p>The assignment allowed me to combine the original folders and redo the split for supervised learning. That is the split used for the classifier and CNN results in their tabs. This random check sampled from the original test folder instead, so 40 of these 50 images had been used to train the supervised models. I use this table to look at their behavior and timing on the same inputs. It is not another test on 50 unseen plates.</p>

      <h3>Random comparison on 50 images</h3>
      <div className="model-table-scroll"><table>
        <thead><tr><th>Model</th><th>Correct good/bad</th><th>Correct class</th><th>Defects caught</th><th>Good plates rejected</th><th>Mean prediction time</th></tr></thead>
        <tbody>
          <tr><th>One class SVM (e1e89a02)</th><td>48/50 (96%)</td><td>Binary only</td><td>35/37 (94.6%)</td><td>0/13</td><td>445.5 ms</td></tr>
          <tr><th>SVM classifier (3437247f)</th><td>48/50 (96%)</td><td>48/50 (96%)</td><td>35/37 (94.6%)</td><td>0/13</td><td>365.5 ms</td></tr>
          <tr><th>Standard whole plate ResNet</th><td>49/50 (98%)</td><td>49/50 (98%)</td><td>36/37 (97.3%)</td><td>0/13</td><td>45.5 ms</td></tr>
          <tr><th>Weighted whole plate ResNet</th><td>49/50 (98%)</td><td>49/50 (98%)</td><td>36/37 (97.3%)</td><td>0/13</td><td>46.0 ms</td></tr>
        </tbody>
      </table></div>

      <p>The four class patch CNN was also promising, particularly because it could show which regions looked defective, but it was much slower. In an earlier comparison of 10 images, it classified all 10 images correctly but took about 14.3 seconds per image. The standard whole plate ResNet also classified those same 10 images correctly and took about 42 ms per image. Both models predict four classes. The patch model takes longer because it runs over many overlapping regions and then combines the predictions. I like the localized predictions, but that cost seems too high for my first choice of a model to inspect every plate. That earlier run mixed train and test images, and the patch model was not included in the 50 image run above. Its regional predictions were useful, but I did not establish an accuracy advantage over the whole plate model in these comparisons.</p>

      <p>From a business perspective, I would care more about a defect passing inspection than a good plate being sent back for another check. The standard ResNet still missed one scratch image after a rotation of 45 degrees, so 98% accuracy does not mean that this problem is solved. My initial choice would be to use the whole plate model for fast inspection and investigate whether a slower patch model or manual review could help with uncertain cases. I would need to test that review rule before assuming it catches the defects that the first model misses.</p>

      <p>For the four day project, this was enough for me to choose which model I would keep working with. I used the evaluation results to guide changes along the way, so even the supervised holdout influenced the final approach. My next step would be to keep the model and preprocessing fixed and try a separate set of plates I had not inspected before. I would focus on how many scratches still pass as good and check the inspection time against the speed of the actual line.</p>
    </div>
  </details>
}
