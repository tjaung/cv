# Run the app with Docker

Start Docker Desktop, then run this from the repository root:

```sh
docker compose up --build -d
```

- Client: http://localhost:5173
- FastAPI docs: http://localhost:8000/docs

The server image installs the local CV package with its CNN dependencies. CV is
an imported Python library, so it runs inside the server rather than in a third
container. The client builds once and is served by Nginx, which forwards `/api/`
requests to FastAPI. The client starts after the server health check passes.

Your dataset must already be present in `server/anomaly_dataset/`. It is mounted
read-only. Results and model files are written back to `CV/results/` and
`artifacts/` on your machine and survive container removal. These large folders
are excluded from the image build. Pretrained ResNet downloads are cached in a
Docker volume and happen on first use when needed.

```sh
docker compose logs -f            # View logs
docker compose down               # Stop and remove the containers
```

After changing client, server, CV code, or writeup images, run the startup
command again to rebuild. Recreating the server stops its running jobs and
clears models loaded in RAM, but keeps files saved to disk. There is no automatic
source reload in these containers.

If the local app already uses the default ports, choose other host ports:

```sh
CLIENT_PORT=5174 SERVER_PORT=8001 docker compose up --build -d
```

This setup uses CPU PyTorch. Docker on macOS does not use the Mac's MPS backend,
so CNN training may be slower than running the server directly on your Mac.
FastAPI uses one worker because background jobs and loaded model instances are
held in process memory. The shared-memory allocation supports PyTorch data
loader workers.
