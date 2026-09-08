# Metal Plate CV Tristar assignment

A lab notebook for exploring metal plate images, preprocessing, features, and model results.

## 1. Install the prerequisites

You need Git and Docker with the Compose plugin. On Ubuntu, install Git with:

```bash
sudo apt update
sudo apt install -y git
```

Follow the [Docker Engine installation guide](https://docs.docker.com/engine/install/) for your machine, which includes the Compose plugin.

Check that Docker is running and accessible:

```bash
docker info
docker compose version
```

## 2. Clone the repository

```bash
git clone https://github.com/tjaung/cv.git
cd cv
```

Run the remaining commands from this directory, where `compose.yaml` is located.

## 3. Add the dataset

The dataset is not included in Git. Copy your extracted `anomaly_dataset` contents into the server folder. Replace the example source path with your dataset location:

```bash
mkdir -p server/anomaly_dataset
cp -a /path/to/anomaly_dataset/. server/anomaly_dataset/
mkdir -p CV/results/data CV/results/models artifacts
```

The expected structure is:

```text
server/anomaly_dataset/
  metal_plate/
    train/
      good/
    test/
      good/
      major_rust/
      scratches/
      total_rust/
    ground_truth/
      major_rust/
      scratches/
      total_rust/
```

Other datasets can sit beside `metal_plate`. Saved models and results are also excluded from Git. Copy them into `CV/results/` if supplied separately, or train models through the app.

## 4. Start everything

```bash
docker compose up --build -d
```

This builds the client and server and installs the CV package inside the server container. You do not need to install Python, Node.js, or the CV dependencies on your host. The first build may take a few minutes.

Check the status or follow the logs:

```bash
docker compose ps
docker compose logs -f
```

Press `Ctrl+C` to stop following the logs. The containers keep running.

## 5. Open the app

- [Lab notebook](http://localhost:5173)
- [FastAPI documentation](http://localhost:8000/docs)

Start with Overview, then use the tabs to explore the dataset and experiments. In the final Comparison and Business Use tab, select saved models and click **Load models** before running predictions.

Results are saved to `CV/results/` and temporary model artifacts to `artifacts/` on your host. Docker uses CPU for model training and prediction.

## Stop or rebuild

Stop the app without deleting saved results:

```bash
docker compose down
```

After changing the code, rebuild and start it again:

```bash
docker compose up --build -d
```

If the default ports are already in use:

```bash
CLIENT_PORT=5174 SERVER_PORT=8001 docker compose up --build -d
```

Then open http://localhost:5174 instead. See [DOCKER.md](DOCKER.md) for more details.
