import kagglehub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# dataset: https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset
DATA_PATH = kagglehub.dataset_download("janstaffa/euro-coins-dataset")


def detect_circle(img_gray):
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))[0, :]
        return circles[0]  # x, y, r xDD 
    return None

# GRAYSCALE I HRV
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    circle = detect_circle(gray)
    if circle is None:
        return None, None

    x, y, r = circle
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    # średnie wartości HSV w masce
    mean_h = np.mean(hsv[:,:,0][mask == 255])
    mean_s = np.mean(hsv[:,:,1][mask == 255])
    mean_v = np.mean(hsv[:,:,2][mask == 255])

    # kontur + okrągłość
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    roundness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

    features = {
        "radius": r,
        "roundness": roundness,
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v
    }

    return circle, features


def classify_coin(features):
    r = features["radius"]
    h = features["mean_h"]

    if r < 60:
        return "1 cent"
    elif r < 70:
        return "2 cent"
    elif r < 85:
        return "10 cent"
    elif r < 95 and h < 40:
        return "1 euro"
    else:
        return "2 euro"


images = glob.glob(os.path.join(DATA_PATH, "**", "*.jpg"), recursive=True)


for path in images[:10]:
    img = cv2.imread(path)
    circle, feats = extract_features(img)

    if feats is None:
        print("Brak wykrytego okręgu")
        continue

    label = classify_coin(feats)

    # Wizualizacja
    x, y, r = circle
    vis = img.copy()
    cv2.circle(vis, (x, y), r, (0, 255, 0), 3)
    cv2.putText(vis, label, (x-50, y-r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} – promień: {feats['radius']}")
    plt.show()


print("DATA_PATH:", DATA_PATH)
print("Pliki:", os.listdir(DATA_PATH))


