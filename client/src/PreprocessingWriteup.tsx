import colorHistograms from '../../image-1.png'
import grayscaleHistograms from '../../image.png'

export default function PreprocessingWriteup() {
  return <details className="panel preprocessing-writeup" open>
    <summary>Preprocessing Pipeline</summary>
    <article className="preprocessing-writeup-content">
      <p>I ran a couple different tests, verifying visually. I ended up landing on this pipeline:</p>
      <div className="writeup-pipelines">
        <section><h3>Segment out the plate</h3><ol>
          <li>Greyscale (Converts the image to brightness values so segmentation can focus on intensity rather than color.)</li>
          <li>Gaussian blur (Smooths small intensity variations and noise before detecting the plate outline.)</li>
          <li>Sobel kernel edge darkening (Detects and darkens edges to strengthen the plate outline before thresholding.)</li>
          <li>ISODATA Algorithm (Finds a threshold to make a binary image, then adds 20 grayscale intensity levels to help retain brighter plate pixels in the foreground when the background is bright.)</li>
          <li>Morphological processing (Fills enclosed holes in the foreground, then applies one dilation to expand it and one erosion to trim it back.)</li>
          <li>Segment out the plate (Keeps the largest foreground region, erodes its border to remove background, and applies the mask to the original color image.)</li>
        </ol></section>
        <section><h3>Pre Process the plate</h3><ol>
          <li>Plate Greyscale (Converts the segmented plate to brightness values for glare detection.)</li>
          <li>Glare threshold (Marks plate pixels brighter than the glare threshold as regions to repair.)</li>
          <li>Fill glare mask with surrounding colors (Replaces each glare region with the median color of nearby non-glare plate pixels.)</li>
          <li>Boundary blend (Softens the transition between the filled glare regions and the surrounding plate.)</li>
          <li>Blur (Smooths a grayscale copy of the repaired plate before local contrast enhancement.)</li>
          <li>CLAHE (Enhances local contrast in small image regions to amplify rust and scratch defects.)</li>
          <li>Contrast / final plate (Increases contrast by 1.5 around the midpoint, then combines that brightness with the repaired color channels for color feature extraction.)</li>
        </ol></section>
      </div>
      <p>{"These two parts are used for my models that I try out. The full pipeline with manually created features is for the classical models for anomaly detection and classification. The CNN approach only uses up until the segmented out plate, followed by the ResNet input transforms. The reason for this is because I wanted to experiment with CNN to see what kind of features it would learn."}</p>
      <p>{"I came to this pipeline after some trial and error."}</p>
      <p>{"I initially had an idea of normalizing colors to a median point. Given a percentage, all colors under mid point would increase luminance by that percentage capping at mid point, and opposite for brighter colors. This did not end up making much of a difference, so I removed it. I also ran some histograms to see if there was a noticeable difference, and there really wasn't enough of a difference to be concerned."}</p>
      <div className="writeup-figures">
        <figure><a href={colorHistograms} target="_blank" rel="noreferrer"><img src={colorHistograms} alt="Color histograms for individual plate images" loading="lazy" /></a><figcaption>Color histograms</figcaption></figure>
        <figure><a href={grayscaleHistograms} target="_blank" rel="noreferrer"><img src={grayscaleHistograms} alt="Grayscale histograms for individual plate images" loading="lazy" /></a><figcaption>Grayscale histograms</figcaption></figure>
      </div>
      <p>{"The main challenge of this was reducing the light reflection from the plate surface. I ended up digging into it and tuning for specific images, which risks fitting the preprocessing too closely to those examples. My concern was that glare would show up as too similar to rust and scratches by using the classical methods. This is something that would need to be revisited after my initial model testing, with a separate final test set that is not used for tuning."}</p>
      <p>{"Why did I segment out the plate? My initial plan was to use PCA and some kind of linear classifier for this problem. The background is always the same in the test and train images, but if given new data, it is not guaranteed that the background will be the same or that there won't be some noise. To reduce the influence of the background, I opted to try segmenting it out so that my analysis would focus on the plates. This still depends on the thresholding and largest-region assumptions working under different lighting and backgrounds. I probably could have just processed the entire image since manufacturing lines are pretty standardized, but I have tried doing thresholding before where slight differences in background color (from lighting) would pick up these small differences, adding noise to binary images."}</p>
      <p>{"Once I had the original segmented plate, I had to again try reducing the glare. I found an article here discussing removing glare from medical images ("}<a href="https://medium.com/@umamahesvari10/removing-glare-from-medical-images-using-patch-based-inpainting-106da16b405a">https://medium.com/@umamahesvari10/removing-glare-from-medical-images-using-patch-based-inpainting-106da16b405a</a>{"). I tried that initially, but it really just colored the glare grey. I tried adapting it so that after the glare mask, it would pick up the colors of the surrounding area and try to color the glare regions as that color. It kind of works, but it introduces some more noise. My reasoning for this is that since the glare is almost impossible to tell what it is underneath, I can't assume that the underside is all good. In the case that the glare is on a rusty part, I would want to capture that, but filling from the surrounding colors cannot recover what is actually hidden underneath. I think I need to adjust the parameters of this, but in the interest of time, I will continue on with it as glare suppression. My concern is that it could hide defects or introduce edges that look like scratches, so I still need to compare performance with and without it."}</p>
    </article>
  </details>
}
