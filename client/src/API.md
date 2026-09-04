# Dataset API

Start FastAPI on port 8000 and run the client with `pnpm dev`.
Vite forwards `/api` requests to FastAPI during development.

```tsx
import { getImageSets, getImages, getImageUrl, getImage, getTestGroundTruth } from './api'

const sets = await getImageSets() // { bracket_black: ['ground_truth', 'test', 'train'], ... }
const names = await getImages('bracket_black', 'test') // ['good/000.png', 'hole/000.png', ...]
const src = getImageUrl('bracket_black', 'test', 'hole/000.png')
const blob = await getImage('bracket_black', 'test', 'hole/000.png')
const pair = await getTestGroundTruth('bracket_black', 'hole', '000.png')
// Display pair.test.data_url and pair.ground_truth?.data_url as image src values.
// ground_truth is null for good images. Failed requests throw an Error.
```

For production, configure your web server to forward `/api` to FastAPI, stripping
the `/api` prefix, or set `VITE_API_BASE_URL` at build time to your server URL.
An API hosted on a different origin needs CORS configured on the server.
