import hsvSpace from '../../../../images/image-3.png'
import labSpace from '../../../../images/image-2.png'

export default function FeaturesWriteup() {
  return <details className="panel preprocessing-writeup" open>
    <summary>Feature Selection</summary>
    <article className="preprocessing-writeup-content">
      <p>The features I ended up choosing for my classifier models were</p>
      <ul>
        <li>Sobel Edge Features (gradients, magnitude, direction)</li>
        <li>HOG (for picking up local patterns of edge strengths and orientations, hopefully for scratches)</li>
        <li>LAB histogram (To separate out good plates with glare from bad plates with rust and scratches)</li>
        <li>Frangi filter (to emphasize bright and dark ridge-like patterns that might be scratches)</li>
      </ul>
      <p>I initially planned out only two features: Sobel edge features (including x, y gradients, magnitude, direction) and a LAB histogram. I chose these two because I thought that they could detect patterns and colors well while accounting for potential glare issues. Sobel edge might pick up the clear separations of major rust, spongy patterns of total rust, and long lines in the same direction for scratches.</p>
      <p>Sobel edge detection will probably also pick up glare though, so I hope that color distributions can help differentiate the glare from scratches and rust, as the glare is often close to white, while rust tends to be colored and scratches can be bright or dark depending on the lighting. In LAB space, L represents lightness, a runs from green to red, and b runs from blue to yellow. Neutral white glare would have high L with a and b near neutral, while rust might shift toward red and yellow. I tried using HSV values too, but I saw less visible separation between classes in those histograms than in LAB.</p>
      <h3>Color spaces</h3>
      <div className="writeup-figures">
        <figure><a href={hsvSpace} target="_blank" rel="noreferrer"><img src={hsvSpace} alt="HSV color space showing hue, saturation and value" loading="lazy" /></a><figcaption>HSV space</figcaption></figure>
        <figure><a href={labSpace} target="_blank" rel="noreferrer"><img src={labSpace} alt="LAB color space showing lightness and the green-red and blue-yellow axes" loading="lazy" /></a><figcaption>LAB space</figcaption></figure>
      </div>
      <p>After my initial model testing, I found that these two features were not enough. My initial models did pretty well in classifying good plates and rust, but pretty much failed completely at detecting scratches. My preprocessing also did not bring out scratches as much as I would have wanted, so I wanted features that could pick up their local patterns better. I added HOG because it uses the Sobel gradients to summarize the strength and orientation of edges in small regions. My hope was that this would help capture the repeated lines in scratches. I also added a Frangi filter to bring out thin, ridge-like patterns at different sizes, although glare and other structures can show up too.</p>
      <p>There were some other tests that failed. I thought local binary patterns could potentially find scratches as well, but I did not see strong signals in the examples I inspected. Trying that did give me the idea of doing the localized region predictions.</p>
      <p>I also initially tried Canny edge detection, but I wanted to keep more information about the edges than its binary output gave me. Since scratches often form lines running in the same direction, I chose Sobel so I could use both the strength and direction of the gradients as features.</p>
    </article>
  </details>
}
