[project_readme.md](https://github.com/user-attachments/files/35048896/project_readme.md)
# Automated Waste Sorter (`IT254-5Guys-proj`)

This project contains the core Python pipeline and configuration files for an automated waste sorter powered by Liquid AI's `lfm3-vl-2b` vision-language model on a Raspberry Pi 5.

## File Overview

* **`waste_sorter.py`** — The main Python application. It captures an image frame (from a connected camera, a provided file, or a synthetic fallback), downscales and encodes it, queries `lfm3-vl-2b` for a structured JSON waste classification (`recycling`, `landfill`, or `compost`), and actuates a servo diverter on GPIO 18 (with a headless simulation fallback if no GPIO hardware is present).
* **`requirements.txt`** — Lists the Python dependencies required by the pipeline (`openai`, `pillow`, `opencv-python`, `numpy`, `python-dotenv`, `gpiozero`, and `rpi-lgpio`).
* **`.env.example`** — A safe environment variable template showing the required configuration keys (`LIQUID_API_KEY` and `LIQUID_BASE_URL`) without containing any actual credentials.
* **`.gitignore`** — Configures Git to ignore local secret files (`.env`), Python virtual environments, bytecode caches, logs, and temporary test images.
