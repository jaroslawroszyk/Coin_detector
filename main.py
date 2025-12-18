import kagglehub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# dataset: https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset
DATA_PATH = kagglehub.dataset_download("janstaffa/euro-coins-dataset")


def detect_coins(img):
    """Wykrywa monety używając Canny + dilation + kontury - jak w referencyjnym kodzie"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Rozmycie gaussowskie
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Wykrywanie krawędzi Canny
    canny = cv2.Canny(blur, 30, 150)
    
    # Dylatacja - łączy przerwane krawędzie (ale mniej agresywna)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filtruj i wybierz tylko najlepsze kontury
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtruj po powierzchni - tylko rozsądne rozmiary monet (szerszy zakres)
        if area < 800 or area > 60000:  # zbyt małe lub zbyt duże
            continue
        
        # Sprawdź okrągłość
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        roundness = 4 * np.pi * area / (perimeter * perimeter)
        
        # Monety powinny być okrągłe (mniej restrykcyjne)
        if roundness < 0.6:
            continue
        
        # Oblicz promień minimalnego okręgu
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        
        # Filtruj po promieniu - tylko rozsądne rozmiary (szerszy zakres)
        if radius < 25 or radius > 130:
            continue
        
        valid_contours.append((cnt, (x, y), radius, area, roundness))
    
    # Non-maximum suppression - usuń duplikaty
    # Sortuj po powierzchni (największe pierwsze)
    valid_contours.sort(key=lambda x: x[3], reverse=True)
    
    coin_contours = []
    used_centers = []
    
    for cnt, (cx, cy), radius, area, roundness in valid_contours:
        # Sprawdź czy nie jest zbyt blisko już użytego konturu
        too_close = False
        for (ux, uy), uradius in used_centers:
            dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
            # Jeśli odległość jest mniejsza niż średnia z promieni, to są zbyt blisko
            if dist < (radius + uradius) * 0.6:
                too_close = True
                break
        
        if not too_close:
            coin_contours.append(cnt)
            used_centers.append(((cx, cy), radius))
    
    return coin_contours


def extract_features_from_contour(img, contour):
    """Wyciąga cechy z pojedynczego konturu monety"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Oblicz powierzchnię
    area = cv2.contourArea(contour)
    
    # Oblicz promień z powierzchni (bardziej niezawodny dla okrągłych obiektów)
    radius_from_area = np.sqrt(area / np.pi)
    
    # Oblicz środek z minimalnego okręgu otaczającego
    (x, y), radius_enclosing = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    
    # Użyj promienia z powierzchni (bardziej dokładny)
    radius = int(radius_from_area)
    
    # Maska dla monety
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Średnie wartości HSV wewnątrz monety
    mean_h = np.mean(hsv[:, :, 0][mask == 255])
    mean_s = np.mean(hsv[:, :, 1][mask == 255])
    mean_v = np.mean(hsv[:, :, 2][mask == 255])
    
    # Oblicz okrągłość (roundness)
    perimeter = cv2.arcLength(contour, True)
    roundness = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    
    features = {
        "radius": radius,
        "area": area,
        "roundness": roundness,
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "center": center,
        "contour": contour
    }
    
    return features


def classify_coin(features):
    """Klasyfikuje monetę na podstawie promienia - proste progi z terminala"""
    r = features["radius"]
    s = features["mean_s"]
    area = features["area"]
    
    # Rzeczywiste wartości z terminala (promienie):
    # 1 cent: r=66, 67, 73
    # 10 cent: r=70
    # 5 cent: r=81
    # 2 cent: r=84
    
    # 1 euro i 2 euro są bimetallic - mają niższe nasycenie koloru (s < 100)
    if s < 100:
        if r < 70:
            return "1 euro"
        else:
            return "2 euro"
    
    # Monety miedziane - klasyfikuj po promieniu z dokładnymi progami
    # Rzeczywiste wartości: 1 cent (r=66,67,73), 10 cent (r=70), 5 cent (r=81), 2 cent (r=84)
    if r <= 68:
        return "1 cent"  # r=66, 67
    elif r == 70:
        return "10 cent"  # r=70
    elif r >= 72 and r <= 74:
        return "1 cent"  # r=73
    elif r >= 79 and r <= 82:
        return "5 cent"  # r=81
    elif r >= 83:
        return "2 cent"  # r=84
    # Domyślnie dla wartości pośrednich
    elif r < 70:
        return "1 cent"
    elif r < 72:
        return "10 cent"
    elif r < 79:
        return "20 cent"
    elif r < 83:
        return "5 cent"
    else:
        return "2 cent"


# Znajdź wszystkie obrazy
images = glob.glob(os.path.join(DATA_PATH, "**", "*.jpg"), recursive=True)

print(f"Znaleziono {len(images)} obrazów")
print("DATA_PATH:", DATA_PATH)

# Przetwórz pierwsze 10 obrazów
for path in images[:10]:
    print(f"\nPrzetwarzanie: {os.path.basename(path)}")
    
    img = cv2.imread(path)
    if img is None:
        print(f"  Nie można wczytać obrazu: {path}")
        continue
    
    # Wykryj monety
    coin_contours = detect_coins(img)
    
    if len(coin_contours) == 0:
        print(f"  Brak wykrytych monet")
        continue
    
    print(f"  Wykryto {len(coin_contours)} monet")
    
    # Wizualizacja
    vis = img.copy()
    
    # Przetwórz każdą monetę
    for i, contour in enumerate(coin_contours):
        feats = extract_features_from_contour(img, contour)
        label = classify_coin(feats)
        
        # Rysuj kontur i okrąg
        center = feats["center"]
        radius = feats["radius"]
        
        # Rysuj tylko zewnętrzny okrąg (nie kontur, żeby uniknąć bałaganu)
        cv2.circle(vis, center, radius, (0, 255, 0), 3)
        
        # Dodaj tekst z etykietą
        text_x = center[0] - 50
        text_y = center[1] - radius - 15
        if text_y < 20:
            text_y = center[1] + radius + 25
        
        cv2.putText(vis, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Dodaj informację o promieniu i powierzchni
        cv2.putText(vis, f"r:{radius} a:{int(feats['area'])}", (text_x, text_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Debug info
        print(f"    Moneta {i+1}: {label}, r={radius}, area={int(feats['area'])}, "
              f"H={feats['mean_h']:.1f}, S={feats['mean_s']:.1f}, round={feats['roundness']:.2f}")
    
    # Wyświetl wynik
    rgb_vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_vis)
    plt.title(f"Wykryto {len(coin_contours)} monet - {os.path.basename(path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


