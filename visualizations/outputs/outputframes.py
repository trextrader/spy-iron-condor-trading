import time
import cv2
import numpy as np
import os
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# -----------------------------
# Output directory
# -----------------------------
OUT = Path("frames")
OUT.mkdir(exist_ok=True)

# -----------------------------
# Chromium options (Lightning-safe)
# -----------------------------
opts = Options()
opts.binary_location = "/usr/bin/chromium-browser"

opts.add_argument("--headless")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-software-rasterizer")
opts.add_argument("--disable-extensions")
opts.add_argument("--disable-background-networking")
opts.add_argument("--disable-sync")
opts.add_argument("--disable-default-apps")
opts.add_argument("--remote-debugging-port=9222")
opts.add_argument("--window-size=1920,1080")

# -----------------------------
# Selenium 4 driver setup
# -----------------------------
service = Service("/usr/bin/chromedriver")

driver = webdriver.Chrome(
    service=service,
    options=opts
)

# -----------------------------
# Load your HTML file
# -----------------------------
html_path = os.path.abspath("neuralorgan_cathedral.html")
driver.get(f"file://{html_path}")

# -----------------------------
# Capture loop
# -----------------------------
N_FRAMES = 30
CAPTURE_INTERVAL = 0.2  # seconds

for i in range(1, N_FRAMES + 1):
    png = driver.get_screenshot_as_png()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(OUT / f"frame_{i:04d}.png"), img)
    time.sleep(CAPTURE_INTERVAL)

driver.quit()
