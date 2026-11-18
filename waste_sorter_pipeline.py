import os
import json
import time
import base64
import argparse
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Environment Variables
ENV_PATH = Path("/opt/waste-sorter/.env")
if not ENV_PATH.exists():
    ENV_PATH = Path(".env")
load_dotenv(ENV_PATH)

API_KEY = os.getenv("LIQUID_API_KEY", "")
BASE_URL = os.getenv("LIQUID_BASE_URL", "https://api.liquid.ai/v1")

if not API_KEY:
    print("[WARNING] No LIQUID_API_KEY found in environment. API calls will fail.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# 2. Hardware Actuation Interface (with graceful headless fallback)
SERVO_PIN = 18
SERVO_ANGLES = {
    "recycling": -1.0,  # Full Left (Chute 1)
    "landfill":   0.0,  # Center Neutral (Chute 2)
    "compost":    1.0   # Full Right (Chute 3)
}

try:
    from gpiozero import Servo
    from gpiozero.pins.lgpio import LGPIOFactory
    servo = Servo(
        SERVO_PIN,
        pin_factory=LGPIOFactory(),
        min_pulse_width=0.5 / 1000,
        max_pulse_width=2.5 / 1000
    )
    HARDWARE_GPIO_AVAILABLE = True
except Exception as e:
    HARDWARE_GPIO_AVAILABLE = False
    print(f"[INFO] Running in simulated GPIO mode ({e})")


def move_diverter(target_bin: str):
    target = target_bin.lower().strip()
    angle = SERVO_ANGLES.get(target, 0.0)
    print(f"[ACTUATOR] Routing to -> {target.upper()} (Angle Position: {angle})")

    if HARDWARE_GPIO_AVAILABLE:
        try:
            servo.value = angle
            time.sleep(1.2)  # Allow item to drop
            servo.value = SERVO_ANGLES["landfill"]  # Reset
            time.sleep(0.3)
        except Exception as e:
            print(f"[HARDWARE ERROR] Servo write failed: {e}")
    else:
        print(f"[SIMULATED SERVO] Moved to {angle}, held 1.2s, reset to neutral.")


# 3. Vision & Frame Handling
def get_camera_frame():
    """Attempts capture from USB webcam or libcamera; falls back to synthetic test frame."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame

    print("[CAMERA] Physical camera not detected. Generating synthetic test item...")
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(dummy, (180, 120), (330, 420), (240, 240, 240), -1)
    cv2.rectangle(dummy, (220, 60), (290, 120), (50, 180, 50), -1)
    return dummy


def prepare_b64_image(frame):
    """Downscales to 512x512 to minimize token usage and base64 encodes."""
    resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")


# 4. Liquid AI Classification
def classify_with_liquid_ai(frame):
    b64_data = prepare_b64_image(frame)

    system_instruction = (
        "You are an automated physical waste sorting sensor. "
        "Analyze the item in the frame and determine if it is recyclable, compostable, or landfill trash. "
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "material": "<detected material>",\n'
        '  "recyclable": <true or false>,\n'
        '  "bin": "<recycling | landfill | compost>",\n'
        '  "confidence": <float between 0.0 and 1.0>\n'
        "}"
    )

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="lfm3-vl-2b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
                    ]
                }
            ],
            max_tokens=80,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        latency = (time.time() - start_time) * 1000
        content = response.choices[0].message.content
        data = json.loads(content)
        data["latency_ms"] = round(latency, 1)
        return data

    except Exception as e:
        print(f"[API ERROR] Liquid AI call failed: {e}")
        return {
            "material": "unknown_or_error",
            "recyclable": False,
            "bin": "landfill",
            "confidence": 0.0,
            "error": str(e)
        }


# 5. Main Execution Loop
def run_cycle(image_path=None):
    if image_path and os.path.exists(image_path):
        print(f"[INPUT] Reading provided file: {image_path}")
        frame = cv2.imread(image_path)
    else:
        frame = get_camera_frame()

    print("[INFERENCE] Sending frame to Liquid AI (lfm3-vl-2b)...")
    verdict = classify_with_liquid_ai(frame)
    print(f"[RESULT] {json.dumps(verdict, indent=2)}")

    move_diverter(verdict.get("bin", "landfill"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liquid AI Waste Sorter Pipeline")
    parser.add_argument("--image", type=str, help="Path to local image file to classify instead of camera")
    parser.add_argument("--loop", action="store_true", help="Run continuously with 5-second sensor interval")
    args = parser.parse_args()

    if args.loop:
        print("[LOOP] Starting automated continuous sorting. Press Ctrl+C to stop.")
        try:
            while True:
                run_cycle(args.image)
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Exiting sorting loop.")
    else:
        run_cycle(args.image)