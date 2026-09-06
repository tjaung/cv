# Preprocessing Pipeline

I ran a couple different test, verifying visually. I ended up landing on this pipeline:

- Segment out the plate
    - Greyscale
    - Gaussian blur
    - Sobel kernel edge darkening
    - ISODATA Algorithm
    - Morphological processing
    - Segment out the plate

- Pre Process the plate
    - Plate Greyscale
    - Glare threshold
    - Fill glare mask with surrounding colors
    - Blur
    - CLAHE

I came to this pipeline after some trial and error. Some ideas that did not end up sticking:

Normalizing colors. I initially had an idea of normalizing colors to a median point. Given a percentage, all colors under mid point would increase luminance by that percentage capping at mid point, and opposite for brighter colors. This did not end up making much of a difference, so I removed it. I also ran some histograms to see if there was a noticeable difference, and there really wasn't enough of a difference to be concerned.

I also tried using contrast filters to reduce glare, but this too did not end up helping much. The glare (particularly for 028.png) is too much for simple contrast filters. 

The main challenge of this was reducing the light reflection from the plate surface. i ended up digging into it and trying to overfit for specific images. My concern was that glare would show up as too similar to rust and scratches by using the classical methods. I think this is something that I will need to come back to after my initial model testing.

Why did I segment out the plate? My initial plan was to use PCA and some kind of linear classifier for this problem. The background is always the same in the test and train images, but if given new data, it is not guarenteed that the background will be the same or that there wont be some noise. In order to keep this model general, I opted to try segmenting it out so that my analysis would capture just the variance of the plates in case of other noise that could come up. I probably could have just processed the entire image since manufacturing lines are pretty standardized, but I have tried doing thresholding before where slight differences in background color (from lighting) would pick up these small differences, adding noise to binary images.

Once I had the original segmented plate, I had to again try reducing the glare. I found an article here discussing removing glare from medical images (https://medium.com/@umamahesvari10/removing-glare-from-medical-images-using-patch-based-inpainting-106da16b405a). I tried that initially, but it really just colored the glare grey. I tried adapting it so that after the glare mask, it would pick up the colors of the surrounding area and try to color the glare regions as that color. It kind of works, but it introduces some more noise. My reasoning for this is that since the glare is almost impossible to tell what it is underneath, I can't assume that the underside is all good. In the case that the glare is on a rusty part, I would want to capture that. I think I need to adjust the parameters of this, but in the interest of time, I will continue on and hope that the patterns of glare are different enough from scratches and rust, but honestly, I think that this is going to reduce model accuracy and pass more defects than I would like.

### Planned features

I think that I am going to try running some edge detection and color histograms as two features for now. The edge detections can pick up large patterns like rust and scratches (unfortunately, glare too). I hope that picking up color distributions can help differentiate the glare from scratches and rust, as the glare is usually a pure white, while rust and scratches are darker. In order to do this, I think using an HSV or LAB type scales might be good, since that can pick up the bright light in the good training images. HSV could have high values for hue and saturation for glare patterns while rust and scratches would have lower values. LAB space would probably show high luminance over these areas and lower blue values, while rust and scratches would indicate higher red values. My intuition says that LAB values would show the variance a little better.

![alt text](image-3.png)
![alt text](image-2.png)
https://www.youtube.com/watch?v=MmBBVTniWFg