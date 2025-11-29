# Project Overview

This project contains a Telegram bot that converts video files into animated GIFs. The bot is written in Python, containerized using Docker, and designed for deployment on a Kubernetes cluster.

The core logic resides in `bot/main.py`. It uses the `python-telegram-bot` library to interact with the Telegram API. Video processing is handled by the `ffmpeg` command-line tool, with `gifsicle` and `imagemagick` used for optimizing the resulting GIFs.

The application is configured to be deployed in a Kubernetes cluster, with manifests managed by Kustomize. A GitHub Actions workflow is set up to automatically build and publish the Docker image to Docker Hub.

## Building and Running

### Prerequisites

*   Docker
*   `kubectl`
*   `kustomize`

### Building the Docker Image

To build the Docker image locally, run the following command from the project root:

```bash
docker build -t video2gif-bot:local ./bot
```

### Running with Kubernetes

1.  **Create the secret:**

    The bot requires a Kubernetes secret to store the Telegram bot token. Create a secret with the token:

    ```bash
    kubectl create secret generic video2gif-bot-secrets --from-literal=tg_video2gif_bot_token='YOUR_TELEGRAM_BOT_TOKEN' -n bots
    ```

    *Note: The default namespace is `bots`. You may need to create it first: `kubectl create namespace bots`*

2.  **Apply the Kustomize configuration:**

    To deploy the bot to your Kubernetes cluster, apply the Kustomize configuration from the `k8s/overlays` directory:

    ```bash
    kubectl apply -k k8s/overlays
    ```

    This will create the `Deployment` and `ServiceAccount` in the `bots` namespace.

## Development Conventions

*   The Python code is located in the `bot/` directory.
*   Dependencies are managed with `pip` and are listed in `bot/requirements.txt`.
*   The Docker image is defined in `bot/Dockerfile`. It uses a slim Python base image and includes `ffmpeg`, `gifsicle`, and `imagemagick`.
*   Kubernetes manifests are organized using Kustomize, with a `base` configuration and an `overlays` directory for environment-specific customizations.
*   CI/CD is handled by GitHub Actions. The workflow in `.github/workflows/dockerhub.yaml` automatically builds and pushes the Docker image to Docker Hub on pushes to the `main` branch.
