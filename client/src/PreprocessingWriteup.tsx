import colorHistograms from '../../image-1.png'
import grayscaleHistograms from '../../image.png'
import hsvSpace from '../../image-3.png'
import labSpace from '../../image-2.png'

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
          <li>Segment out the plate (Keeps the largest foreground region, erodes its border to remove background, and applies the mask to the normalized color image.)</li>
        </ol></section>
        <section><h3>Pre Process the plate</h3><ol>
          <li>Plate Greyscale (Converts the segmented plate to brightness values for glare detection.)</li>
          <li>Glare threshold (Marks plate pixels brighter than the glare threshold as regions to repair.)</li>
          <li>Fill glare mask with surrounding colors (Replaces each glare region with the median color of nearby non-glare plate pixels.)</li>
          <li>Blur (Smooths a grayscale copy of the repaired plate before local contrast enhancement.)</li>
          <li>CLAHE (Enhances local contrast in small image regions while limiting excessive noise amplification.)</li>
        </ol></section>
      </div>
      <p>{"These two steps are used for my models that I try out. The full pipeline with manually created featuers are for the linear classifiers for anomaly detection and classification. The CNN approach only uses up until the segmented out plate. The reason for this is because I wanted to experiment with CNN to see what kind of features it would learn."}</p>
      <p>{"I came to this pipeline after some trial and error. Some ideas that did not end up sticking:"}</p>
      <p>{"Normalizing colors. I initially had an idea of normalizing colors to a median point. Given a percentage, all colors under mid point would increase luminance by that percentage capping at mid point, and opposite for brighter colors. This did not end up making much of a difference, so I removed it. I also ran some histograms to see if there was a noticeable difference, and there really wasn't enough of a difference to be concerned."}</p>
      <div className="writeup-figures">
        <figure><a href={colorHistograms} target="_blank" rel="noreferrer"><img src={colorHistograms} alt="Color histograms for individual plate images" loading="lazy" /></a><figcaption>Color histograms</figcaption></figure>
        <figure><a href={grayscaleHistograms} target="_blank" rel="noreferrer"><img src={grayscaleHistograms} alt="Grayscale histograms for individual plate images" loading="lazy" /></a><figcaption>Grayscale histograms</figcaption></figure>
      </div>
      <p>{"I also tried using contrast filters to reduce glare, but this too did not end up helping much. The glare (particularly for 028.png) is too much for simple contrast filters."}</p>
      <p>{"The main challenge of this was reducing the light reflection from the plate surface. i ended up digging into it and trying to overfit for specific images. My concern was that glare would show up as too similar to rust and scratches by using the classical methods. This is something that would need to be revisited after my initial model testing."}</p>
      <p>{"Why did I segment out the plate? My initial plan was to use PCA and some kind of linear classifier for this problem. The background is always the same in the test and train images, but if given new data, it is not guarenteed that the background will be the same or that there wont be some noise. In order to keep this model general, I opted to try segmenting it out so that my analysis would capture just the variance of the plates in case of other noise that could come up. I probably could have just processed the entire image since manufacturing lines are pretty standardized, but I have tried doing thresholding before where slight differences in background color (from lighting) would pick up these small differences, adding noise to binary images."}</p>
      <p>{"Once I had the original segmented plate, I had to again try reducing the glare. I found an article here discussing removing glare from medical images ("}<a href="https://medium.com/@umamahesvari10/removing-glare-from-medical-images-using-patch-based-inpainting-106da16b405a">https://medium.com/@umamahesvari10/removing-glare-from-medical-images-using-patch-based-inpainting-106da16b405a</a>{"). I tried that initially, but it really just colored the glare grey. I tried adapting it so that after the glare mask, it would pick up the colors of the surrounding area and try to color the glare regions as that color. It kind of works, but it introduces some more noise. My reasoning for this is that since the glare is almost impossible to tell what it is underneath, I can't assume that the underside is all good. In the case that the glare is on a rusty part, I would want to capture that. I think I need to adjust the parameters of this, but in the interest of time, I will continue on and hope that the patterns of glare are different enough from scratches and rust, but honestly, I think that this is going to reduce model accuracy and pass more defects than I would like."}</p>
      <h3>Color spaces</h3>
      <div className="writeup-figures">
        <figure><a href={hsvSpace} target="_blank" rel="noreferrer"><img src={hsvSpace} alt="HSV color space showing hue, saturation and value" loading="lazy" /></a><figcaption>HSV space</figcaption></figure>
        <figure><a href={labSpace} target="_blank" rel="noreferrer"><img src={labSpace} alt="LAB color space showing lightness and the green-red and blue-yellow axes" loading="lazy" /></a><figcaption>LAB space</figcaption></figure>
      </div>
    </article>
  </details>
}
