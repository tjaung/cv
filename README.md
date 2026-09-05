# Metal plate CV

`CV/` is the reusable Python package for preprocessing and individual computer
vision steps. `server/` exposes it through FastAPI; `client/` displays the results.

Install the package and server dependencies from the repository root:

```sh
venv/bin/python -m pip install -e . -r server/requirements.txt
venv/bin/python -m uvicorn server.main:app --reload
```

The editable installation makes `CV` importable from any working directory,
including `server/`, using the same Python environment.

```python
import cv2
from CV.preprocessing import preprocess_plate
from CV.preprocessing.gaussian_blur import apply_gaussian_blur

image = cv2.imread('path/to/image.png')
result = preprocess_plate(image)
plate = result.blended_plate
edges = result.segmented_edges
```

All existing step functions, including inactive alternatives, live under
`CV/preprocessing/`. The processing order and HTTP endpoints are unchanged.
Dataset files remain under `server/anomaly_dataset/`.

For the command-line stage preview, install plotting dependencies:

```sh
venv/bin/python -m pip install -e '.[plot]'
venv/bin/preprocess-plate path/to/image.png --output-dir /tmp/plate-preview
```

Run package and API tests:

```sh
venv/bin/python -m unittest discover -s CV/preprocessing/tests
venv/bin/python -m unittest discover -s server/tests
```

See [server setup and API](server/README.md) for endpoint details.
