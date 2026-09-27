"""Start PuppyCam: python -m puppycam"""

from __future__ import annotations

import logging

import uvicorn

from .config import Settings, prepare_environment
from .dataset import Dataset
from .engine import Engine
from .registry import ModelRegistry
from .server import create_app


def main() -> None:
    prepare_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("puppycam")
    if not settings.cameras:
        log.warning("No cameras configured: set CAM1_URL (and CAM2_URL, ...) in your .env file")
    for cam in settings.cameras:
        log.info("camera %s (%s)", cam.label, cam.slug)

    registry = ModelRegistry(settings.data_dir)
    dataset = Dataset(settings.data_dir / "dataset")
    engine = Engine(settings, registry, dataset).start()
    log.info("expecting %d puppies; running on %s; open http://localhost:%d", settings.expected_puppies,
             engine.device, settings.port)
    uvicorn.run(create_app(engine, settings), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
