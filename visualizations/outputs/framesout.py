import time
import cv2
import numpy as np
import os
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

OUT = Path("frames")
OUT.mkdir(exist_ok=True)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1920,1080")

service = Service(os.path.abspath("chromedriver.exe"))

driver = webdriver.Chrome(
    service=service,
    options=opts
)

html_path = os.path.abspath("NeuralOrgan_Cathedral.html")
driver.get(f"file:///{html_path}")

N_FRAMES = 30
CAPTURE_INTERVAL = 0.2

for i in range(1, N_FRAMES + 1):
    png = driver.get_screenshot_as_png()
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite(str(OUT / f"frame_{i:04d}.png"), img)
    time.sleep(CAPTURE_INTERVAL)

driver.quit()
