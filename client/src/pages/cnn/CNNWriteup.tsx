import Writeup from '../../components/Writeup'

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
