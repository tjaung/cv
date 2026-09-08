"""Shared geometry for segmented-plate patch extraction and previews."""

def patch_boxes(mask, size=64, stride=32, min_coverage=.5):
    if size < 1 or not 1 <= stride <= size:
        raise ValueError('Require positive patch size and stride <= size')
    height, width = mask.shape
    def starts(length):
        last = max(0, length-size)
        return sorted(set(range(0, last+1, stride)) | {last})
    return [dict(left=x, top=y, right=min(x+size, width), bottom=min(y+size, height))
            for y in starts(height) for x in starts(width)
            if (mask[y:y+size, x:x+size] > 0).mean() >= min_coverage]

