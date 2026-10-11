"""
Aplikasi Crop Otomatis Box pada Gambar
Menggunakan OpenCV — deteksi grid multi-metode yang robust
untuk gambar scan, foto, dan form tulisan tangan.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
#  PRE-PROCESSING
# ─────────────────────────────────────────────

def find_grid_line(gray, start_x, end_x, start_y, end_y, axis, target_pos=None):
    """
    Mencari koordinat garis grid (tengah-tengah garis hitam).
    axis: 0 mencari garis horizontal (Y), 1 mencari garis vertikal (X).
    target_pos: koordinat acuan (posisi target yang diharapkan). Jika diberikan,
                akan memilih puncak garis yang paling dekat dengan target_pos
                untuk mencegah lompatan ke garis lain yang lebih tebal/jauh.
    """
    h_local, w_local = gray.shape
    start_x = max(0, int(start_x))
    end_x = min(w_local, int(end_x))
    start_y = max(0, int(start_y))
    end_y = min(h_local, int(end_y))
    
    strip = gray[start_y:end_y, start_x:end_x]
    if strip.size == 0:
        return None
        
    thresh_local = cv2.adaptiveThreshold(strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    
    kernel = np.ones((3,3), np.uint8)
    thresh_local = cv2.dilate(thresh_local, kernel, iterations=1)
    
    if axis == 0:
        profile = np.sum(thresh_local > 0, axis=1)
        offset = start_y
        length = end_x - start_x
    else:
        profile = np.sum(thresh_local > 0, axis=0)
        offset = start_x
        length = end_y - start_y
        
    if len(profile) == 0:
        return None
        
    min_peak = length * 0.15 
    max_val = np.max(profile)
    if max_val < min_peak:
        return None

    if target_pos is not None:
        # Cari semua segmen puncak lokal yang melebihi threshold
        th = max(min_peak, max_val * 0.35)
        peaks = []
        in_peak = False
        p_start = 0
        for idx, val in enumerate(profile):
            if val >= th and not in_peak:
                in_peak = True
                p_start = idx
            elif val < th and in_peak:
                in_peak = False
                p_center = (p_start + idx - 1) // 2
                peaks.append((offset + p_center, np.max(profile[p_start:idx])))
        if in_peak:
            p_center = (p_start + len(profile) - 1) // 2
            peaks.append((offset + p_center, np.max(profile[p_start:])))
            
        if peaks:
            best_pos = min(peaks, key=lambda p: abs(p[0] - target_pos))[0]
            return best_pos
            
    # Default: ambil puncak nilai tertinggi
    peak_idx = np.argmax(profile)
    peak_val = profile[peak_idx]
    if peak_val >= min_peak:
        return offset + peak_idx
    return None

def deskew_image(image):
    """
    Mendeteksi kemiringan gambar (skew angle) berdasarkan garis dominan
    dan memutarnya kembali agar lurus (0 derajat).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # Binarize untuk menonjolkan garis pudar (solusi untuk garis putus-putus)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    
    # Deteksi tepi dengan Canny pada gambar yang sudah di-threshold
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Hitung panjang minimum garis secara dinamis (8% dari dimensi terkecil gambar)
    min_line_len = int(min(h, w) * 0.08)
    min_line_len = max(30, min_line_len)
    
    # Deteksi garis dengan HoughLinesP (gap lebih besar untuk menyambung garis putus)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, 
                            minLineLength=min_line_len, maxLineGap=40)
    
    if lines is None:
        return image, 0.0

    angles = []
    border_margin = int(min(h, w) * 0.08)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Abaikan garis yang terlalu dekat dengan tepi luar gambar (digital boundary/scanner artifact)
        if (x1 < border_margin or x2 < border_margin or 
            y1 < border_margin or y2 < border_margin or
            x1 > w - border_margin or x2 > w - border_margin or
            y1 > h - border_margin or y2 > h - border_margin):
            continue

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalkan sudut ke horizontal/vertikal (-45 s.d +45 derajat)
        if -15 < angle < 15:
            norm_angle = angle
        elif angle > 75:
            norm_angle = angle - 90
        elif angle < -75:
            norm_angle = angle + 90
        else:
            continue
            
        angles.append(norm_angle)

    if not angles:
        return image, 0.0

    # Cari sudut dominan (mode) menggunakan pembulatan ke 0.1 derajat
    from collections import Counter
    rounded_angles = [round(a * 10) / 10 for a in angles]
    counts = Counter(rounded_angles)
    most_common_angle, count = counts.most_common(1)[0]
    
    # Jika sudut dominan sangat dekat dengan 0 (lurus), set skew_angle = 0
    if abs(most_common_angle) < 0.10:
        median_angle = 0.0
    else:
        median_angle = most_common_angle

    # Jika kemiringan 0 atau sangat kecil, jangan diputar (menjaga ketajaman)
    if abs(median_angle) < 0.10:
        return image, 0.0

    # Rotasi gambar
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    # Lakukan rotasi dengan background putih bersih
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    print(f"[DESKEW] Berhasil meluruskan kertas miring. Sudut kemiringan terdeteksi: {median_angle:.2f}°")
    return rotated, median_angle


def preprocess_image(image):
    """Grayscale + bilateral filter + adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral: hapus noise tapi pertahankan tepi garis
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 6
    )
    return gray, thresh


# ─────────────────────────────────────────────
#  ESTIMASI DPI & KONVERSI mm → px
# ─────────────────────────────────────────────

def estimate_dpi(image_shape):
    """Estimasi DPI dengan asumsi kertas A4 (210×297mm)."""
    h, w = image_shape[:2]
    dpi_w = w / (210 / 25.4)
    dpi_h = h / (297 / 25.4)
    return max(72, min(600, (dpi_w + dpi_h) / 2))


def mm_to_px(mm, dpi):
    """Konversi mm ke piksel."""
    return max(1, int(round(mm * dpi / 25.4)))


# ─────────────────────────────────────────────
#  DETEKSI GARIS — MORFOLOGI (multi-scale)
# ─────────────────────────────────────────────

def detect_lines_morph(thresh):
    """
    Deteksi garis H & V dengan morfologi.
    Coba beberapa ukuran kernel dari ketat -> longgar
    agar bisa menangani gambar scan maupun foto.
    """
    h, w = thresh.shape

    # Kandidat panjang kernel: dari 1/3 lebar hingga 1/15 lebar
    # Ditambahkan divisor 10, 12, 15 agar garis pendek/tipis juga terdeteksi
    h_candidates = [w // k for k in [3, 4, 5, 6, 8, 10, 12, 15] if w // k >= 20]
    v_candidates = [h // k for k in [3, 4, 5, 6, 8, 10, 12, 15] if h // k >= 20]

    best_horiz = np.zeros_like(thresh)
    best_vert  = np.zeros_like(thresh)

    # Coba setiap ukuran kernel, ambil yang mendeteksi paling banyak garis
    for kw in h_candidates:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        tmp = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        tmp = cv2.dilate(tmp, np.ones((3, 1), np.uint8), iterations=2)
        if tmp.sum() > best_horiz.sum():
            best_horiz = tmp

    for kh in v_candidates:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
        tmp = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        tmp = cv2.dilate(tmp, np.ones((1, 3), np.uint8), iterations=2)
        if tmp.sum() > best_vert.sum():
            best_vert = tmp

    return best_horiz, best_vert


# ─────────────────────────────────────────────
#  DETEKSI GARIS — HOUGH (fallback)
# ─────────────────────────────────────────────

def detect_lines_hough(gray, image_shape):
    """
    Deteksi garis menggunakan Hough Transform — robust untuk
    gambar yang garisnya tipis atau terputus-putus.
    Mengembalikan (horiz_mask, vert_mask).
    """
    h, w = image_shape[:2]
    edges = cv2.Canny(gray, 30, 120, apertureSize=3)

    # Turunkan minimum panjang garis agar garis tabel yang lebih pendek
    # (karena scan/foto) tetap terdeteksi
    min_line_h = int(w * 0.20)
    min_line_v = int(h * 0.15)

    horiz_mask = np.zeros((h, w), np.uint8)
    vert_mask  = np.zeros((h, w), np.uint8)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=50,
                             minLineLength=min(min_line_h, min_line_v),
                             maxLineGap=30)
    if lines is None:
        return horiz_mask, vert_mask

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.hypot(x2 - x1, y2 - y1)

        if angle < 10 and length >= min_line_h:          # Horizontal
            y_mid = (y1 + y2) // 2
            cv2.line(horiz_mask, (0, y_mid), (w, y_mid), 255, 3)
        elif angle > 80 and length >= min_line_v:         # Vertikal
            x_mid = (x1 + x2) // 2
            cv2.line(vert_mask, (x_mid, 0), (x_mid, h), 255, 3)

    return horiz_mask, vert_mask


# ─────────────────────────────────────────────
#  ESTIMASI KETEBALAN GARIS
# ─────────────────────────────────────────────

def estimate_line_thickness(line_mask, axis):
    """Estimasi ketebalan garis dari mask biner (konservatif, percentile 80)."""
    proj = np.sum(line_mask > 0, axis=axis)
    threshold = proj.max() * 0.2
    widths, in_peak, start = [], False, 0
    for i, v in enumerate(proj):
        if v > threshold and not in_peak:
            in_peak, start = True, i
        elif v <= threshold and in_peak:
            in_peak = False
            widths.append(i - start)
    if not widths:
        return 4
    # Gunakan percentile 80 agar estimasi lebih konservatif
    # (menghindari underestimate yang menyebabkan garis ikut ter-crop)
    return max(3, int(np.percentile(widths, 80)))


# ─────────────────────────────────────────────
#  TEMUKAN POSISI GARIS GRID
# ─────────────────────────────────────────────

def refine_grid_positions(positions, total_length, base_thick=5, min_expected=3):
    """
    Menyaring posisi garis grid agar konsisten (equidistant).
    
    Strategi:
    1. Buang garis yang terlalu dekat dengan tepi gambar (< 3% dari total),
       karena ini biasanya tepi kertas/scanner, bukan garis tabel.
    2. Merge garis yang terlalu berdekatan (< 5% dari total) menjadi
       satu garis di titik tengah. Hitung juga ketebalan fisik gabungan.
    3. Hitung jarak antar garis berurutan.
    4. Cari jarak dominan (spacing paling sering muncul +/- toleransi).
    5. Hanya pertahankan garis yang jarak ke tetangganya konsisten 
       dengan spacing dominan.
    """
    if len(positions) < min_expected:
        return positions, base_thick

    # 1. Buang garis tepi gambar (< 3% dari tepi)
    border_margin = int(total_length * 0.03)
    inner = [p for p in positions if border_margin < p < (total_length - border_margin)]
    
    if len(inner) < min_expected:
        inner = [p for p in positions if p > 5 and p < (total_length - 5)]
    
    if len(inner) < 2:
        return positions, base_thick

    # 2. Merge garis yang berdekatan (max 1.5% dari panjang atau min 15px)
    merge_dist = max(15, int(total_length * 0.015))
    merged = []
    max_thick = base_thick
    group = [inner[0]]
    
    for i in range(1, len(inner)):
        if inner[i] - group[-1] <= merge_dist:
            group.append(inner[i])
        else:
            merged.append(int(np.mean(group)))
            thick = (group[-1] - group[0]) + base_thick
            if thick > max_thick: max_thick = thick
            group = [inner[i]]
            
    merged.append(int(np.mean(group)))
    thick = (group[-1] - group[0]) + base_thick
    if thick > max_thick: max_thick = thick
    inner = merged

    if len(inner) < 2:
        return positions, max_thick

    # 3. Hitung jarak antar garis berurutan
    gaps = [inner[i+1] - inner[i] for i in range(len(inner) - 1)]
    
    if not gaps:
        return positions, max_thick

    # FILTER: Jarak antar sel tabel yang valid minimal ~10% dari panjang total
    # (karena kita mengharapkan 3 kolom atau 4-5 baris)
    # Ini akan mengabaikan jarak sempit akibat coretan/tulisan tangan
    valid_gaps = [g for g in gaps if g > total_length * 0.08]
    if not valid_gaps:
        # Jika tidak ada gap besar, kemungkinan deteksi gagal, fallback ke gap terbesar
        valid_gaps = [max(gaps)]

    # 4. Cari spacing dominan menggunakan histogram
    gap_groups = {}
    for g in sorted(valid_gaps):
        placed = False
        for key in gap_groups:
            if abs(g - key) < key * 0.20:
                gap_groups[key].append(g)
                placed = True
                break
        if not placed:
            gap_groups[g] = [g]
    
    dominant_group = max(gap_groups.values(), key=len)
    dominant_spacing = int(np.median(dominant_group))
    tolerance = dominant_spacing * 0.25  # 25% toleransi
    
    # 5. Filter: Cari rantai (chain) garis yang konsisten
    best_chain = []
    for i in range(len(inner)):
        chain = [inner[i]]
        current = inner[i]
        for j in range(i+1, len(inner)):
            dist = inner[j] - current
            
            if abs(dist - dominant_spacing) <= tolerance:
                chain.append(inner[j])
                current = inner[j]
            elif abs(dist - 2*dominant_spacing) <= tolerance * 1.5:
                # Inferensi garis yang hilang di tengah
                chain.append(current + dominant_spacing)
                chain.append(inner[j])
                current = inner[j]
            elif abs(dist - 3*dominant_spacing) <= tolerance * 2:
                # Inferensi 2 garis yang hilang
                chain.append(current + dominant_spacing)
                chain.append(current + 2*dominant_spacing)
                chain.append(inner[j])
                current = inner[j]
                
        if len(chain) > len(best_chain):
            best_chain = chain
    
    refined = best_chain
    
    if len(refined) < min_expected:
        return (inner, max_thick) if len(inner) >= min_expected else (positions, base_thick)
    
    print(f"    [Refine] {len(positions)} garis -> {len(refined)} (spacing~{dominant_spacing}px, tebal_maks={max_thick}px)")
    return refined, max_thick


def normalize_column_positions(col_pos, expected_cols=3, img_width=None):
    """
    Sederhanakan normalisasi kolom.
    Gunakan batas kiri dan kanan yang terdeteksi, lalu bagi rata.
    Jika rentang terlalu sempit (hilang garis luar), gunakan default aman 10%-90%.
    """
    if len(col_pos) < 2:
        return col_pos

    first = col_pos[0]
    last = col_pos[-1]
    span = last - first

    # Jika rentang terlalu sempit (< 65% lebar), kemungkinan garis luar (kiri/kanan) tidak terdeteksi
    if img_width and span < img_width * 0.65:
        print(f"    [NormCol] Rentang kolom terlalu sempit ({span}px). Menggunakan default 10%-90%.")
        first = int(img_width * 0.1)
        last = int(img_width * 0.9)
        span = last - first

    target_n = expected_cols + 1
    spacing = span / expected_cols
    normalized = [int(round(first + i * spacing)) for i in range(target_n)]
    
    print(f"    [NormCol] {len(col_pos)} garis -> {target_n} equidistant (spacing={spacing:.0f}px, range={first}-{last})")
    return normalized


def find_line_inner_edge(gray, line_pos, span_start, span_end, axis, direction,
                          search_range=40, dark_thresh=200, min_dark_ratio=0.10):
    """
    Dari posisi garis yang terdeteksi, scan tegak lurus ke arah dalam sel
    untuk menemukan di mana garis sesungguhnya berakhir (tepi dalam garis).

    Ini lebih presisi daripada padding statis berdasarkan estimasi ketebalan,
    karena mengukur tepi garis yang sebenarnya secara lokal.

    Args:
        gray: Gambar grayscale
        line_pos: Posisi pusat garis yang terdeteksi (piksel)
        span_start, span_end: Rentang sampling paralel garis
            (kolom untuk garis H, baris untuk garis V)
        axis: 0 = garis horizontal (scan vertikal), 1 = garis vertikal (scan horizontal)
        direction: +1 = scan ke koordinat lebih besar (ke bawah/ke kanan)
                   -1 = scan ke koordinat lebih kecil (ke atas/ke kiri)
        search_range: Jarak maksimum pencarian dari posisi garis (piksel)
        dark_thresh: Piksel di bawah nilai ini dianggap gelap (bagian garis)
        min_dark_ratio: Rasio minimum piksel gelap agar dianggap masih bagian garis

    Returns:
        Koordinat piksel pertama yang bukan garis (tepi dalam)
    """
    h, w = gray.shape

    # Sample bagian tengah 60% dari span untuk menghindari artefak sudut
    span_len = max(1, span_end - span_start)
    s0 = max(0, span_start + int(span_len * 0.20))
    s1 = span_end - int(span_len * 0.20)

    if axis == 0:  # Garis horizontal — scan vertikal
        s1 = min(s1, w)
        if s1 <= s0:
            s0, s1 = max(0, span_start), min(w, span_end)

        if direction > 0:  # Scan ke bawah (tepi atas sel di bawah garis)
            start_y = max(0, line_pos)
            end_y = min(h, line_pos + search_range)
            for y in range(start_y, end_y):
                strip = gray[y, s0:s1]
                if len(strip) == 0:
                    continue
                if np.sum(strip < dark_thresh) / len(strip) < min_dark_ratio:
                    return y
            return end_y
        else:  # Scan ke atas (tepi bawah sel di atas garis)
            start_y = min(h - 1, line_pos)
            end_y = max(-1, line_pos - search_range)
            for y in range(start_y, end_y, -1):
                strip = gray[y, s0:s1]
                if len(strip) == 0:
                    continue
                if np.sum(strip < dark_thresh) / len(strip) < min_dark_ratio:
                    return y
            return max(0, end_y + 1)

    else:  # Garis vertikal — scan horizontal
        s1 = min(s1, h)
        if s1 <= s0:
            s0, s1 = max(0, span_start), min(h, span_end)

        if direction > 0:  # Scan ke kanan (tepi kiri sel di kanan garis)
            start_x = max(0, line_pos)
            end_x = min(w, line_pos + search_range)
            for x in range(start_x, end_x):
                strip = gray[s0:s1, x]
                if len(strip) == 0:
                    continue
                if np.sum(strip < dark_thresh) / len(strip) < min_dark_ratio:
                    return x
            return end_x
        else:  # Scan ke kiri (tepi kanan sel di kiri garis)
            start_x = min(w - 1, line_pos)
            end_x = max(-1, line_pos - search_range)
            for x in range(start_x, end_x, -1):
                strip = gray[s0:s1, x]
                if len(strip) == 0:
                    continue
                if np.sum(strip < dark_thresh) / len(strip) < min_dark_ratio:
                    return x
            return max(0, end_x + 1)


def find_grid_positions(horiz, vert, image_shape, min_gap=25):
    """
    Proyeksikan mask garis untuk menemukan posisi pusat
    setiap garis baris dan kolom.
    """
    h, w = image_shape[:2]

    def peak_centers(proj, min_ratio=0.12, gap=min_gap):
        if proj.max() == 0:
            return []
        threshold = proj.max() * min_ratio
        in_peak, start, peaks = False, 0, []
        for i, val in enumerate(proj):
            if val > threshold and not in_peak:
                in_peak, start = True, i
            elif val <= threshold and in_peak:
                in_peak = False
                center = (start + i) // 2
                if not peaks or (center - peaks[-1]) > gap:
                    peaks.append(center)
        return peaks

    h_proj = np.sum(horiz, axis=1).astype(float)
    v_proj = np.sum(vert,  axis=0).astype(float)

    row_pos = peak_centers(h_proj, gap=min_gap)
    col_pos = peak_centers(v_proj, gap=min_gap)

    # Estimasi ketebalan awal
    h_thick_base = estimate_line_thickness(horiz, axis=1)
    v_thick_base = estimate_line_thickness(vert,  axis=0)

    # Penyaringan konsistensi grid & kalkulasi ketebalan gabungan
    row_pos, h_thick = refine_grid_positions(row_pos, h, base_thick=h_thick_base, min_expected=4)
    col_pos, v_thick = refine_grid_positions(col_pos, w, base_thick=v_thick_base, min_expected=4)

    # Normalisasi posisi kolom agar equidistant (3 kolom sama lebar)
    if len(col_pos) >= 2:
        col_pos = normalize_column_positions(col_pos, expected_cols=3, img_width=w)
        
    return row_pos, col_pos, h_thick, v_thick



# ─────────────────────────────────────────────
#  DETEKSI GRID — GABUNGAN
# ─────────────────────────────────────────────

def detect_grid(gray, thresh, image_shape):
    """
    Deteksi grid lengkap: coba morfologi DAN Hough, pilih yang terbaik.
    Mengembalikan (row_positions, col_positions, h_thickness, v_thickness,
                   horiz_mask, vert_mask).
    """
    # Tahap 1: Morfologi
    horiz_m, vert_m = detect_lines_morph(thresh)
    row_m, col_m, ht_m, vt_m = find_grid_positions(horiz_m, vert_m, image_shape)
    morph_score = (len(row_m) - 1) * (len(col_m) - 1) if len(row_m) >= 2 and len(col_m) >= 2 else 0
    print(f"  [Morph] baris={len(row_m)}, kolom={len(col_m)}, "
          f"tebal H={ht_m}px V={vt_m}px (skor={morph_score})")

    # Tahap 2: Hough — selalu dicoba untuk perbandingan
    horiz_h, vert_h = detect_lines_hough(gray, image_shape)
    row_h, col_h, ht_h, vt_h = find_grid_positions(horiz_h, vert_h, image_shape)
    hough_score = (len(row_h) - 1) * (len(col_h) - 1) if len(row_h) >= 2 and len(col_h) >= 2 else 0
    print(f"  [Hough] baris={len(row_h)}, kolom={len(col_h)}, "
          f"tebal H={ht_h}px V={vt_h}px (skor={hough_score})")

    # Tahap 3: Gabung morfologi + Hough
    horiz_c = cv2.add(horiz_m, horiz_h)
    vert_c  = cv2.add(vert_m,  vert_h)
    row_c, col_c, ht_c, vt_c = find_grid_positions(horiz_c, vert_c, image_shape)
    merge_score = (len(row_c) - 1) * (len(col_c) - 1) if len(row_c) >= 2 and len(col_c) >= 2 else 0
    print(f"  [Merge] baris={len(row_c)}, kolom={len(col_c)} (skor={merge_score})")

    # Pilih metode dengan skor tertinggi (jumlah sel terbanyak)
    best = max(
        ('Morph', morph_score, row_m, col_m, ht_m, vt_m, horiz_m, vert_m),
        ('Hough', hough_score, row_h, col_h, ht_h, vt_h, horiz_h, vert_h),
        ('Merge', merge_score, row_c, col_c, ht_c, vt_c, horiz_c, vert_c),
        key=lambda x: x[1]
    )
    name, score, rows, cols, ht, vt, hz, vr = best
    print(f"  [Best] -> {name} dengan {score} sel")
    return rows, cols, ht, vt, hz, vr


# ─────────────────────────────────────────────
#  BERSIHKAN GARIS SISA DI TEPI SEL (SKEW)
# ─────────────────────────────────────────────

def clean_border_lines(cell_img, scan_depth_ratio=0.10, dark_thresh=160, line_ratio=0.25):
    """
    Mendeteksi dan memotong sisa garis tabel yang masih menempel di tepi sel
    akibat kemiringan gambar (skew). Memindai dari setiap tepi ke dalam
    dan mencari baris/kolom piksel yang didominasi piksel gelap (garis).
    
    Args:
        cell_img: Gambar sel yang sudah di-crop (BGR atau Grayscale)
        scan_depth_ratio: Seberapa dalam pemindaian dari tepi (0.10 = 10% dari dimensi)
        dark_thresh: Piksel di bawah nilai ini dianggap gelap (bagian garis)
        line_ratio: Minimal rasio piksel gelap per baris/kolom untuk dianggap garis
    """
    if cell_img is None or cell_img.size == 0:
        return cell_img
    
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    h, w = gray.shape
    
    if h < 20 or w < 20:
        return cell_img
    
    scan_h = max(4, int(h * scan_depth_ratio))
    scan_w = max(4, int(w * scan_depth_ratio))
    
    # Cek tepi ATAS: pindai baris demi baris dari atas ke bawah
    top_trim = 0
    for y in range(scan_h):
        row = gray[y, :]
        dark_ratio = np.sum(row < dark_thresh) / w
        if dark_ratio > line_ratio:
            top_trim = y + 1
        elif top_trim > 0:
            break  # Sudah lewat garis, berhenti
    
    # Cek tepi BAWAH: pindai dari bawah ke atas
    bottom_trim = h
    for y in range(h - 1, h - scan_h - 1, -1):
        row = gray[y, :]
        dark_ratio = np.sum(row < dark_thresh) / w
        if dark_ratio > line_ratio:
            bottom_trim = y
        elif bottom_trim < h:
            break
    
    # Cek tepi KIRI: pindai kolom demi kolom dari kiri ke kanan
    left_trim = 0
    for x in range(scan_w):
        col = gray[:, x]
        dark_ratio = np.sum(col < dark_thresh) / h
        if dark_ratio > line_ratio:
            left_trim = x + 1
        elif left_trim > 0:
            break
    
    # Cek tepi KANAN: pindai dari kanan ke kiri
    right_trim = w
    for x in range(w - 1, w - scan_w - 1, -1):
        col = gray[:, x]
        dark_ratio = np.sum(col < dark_thresh) / h
        if dark_ratio > line_ratio:
            right_trim = x
        elif right_trim < w:
            break
    
    # Pastikan hasil trim masih cukup besar
    if (bottom_trim - top_trim) < 15 or (right_trim - left_trim) < 15:
        return cell_img  # Tidak jadi trim jika terlalu kecil
    
    # Hanya trim jika ada perubahan signifikan
    if top_trim > 0 or bottom_trim < h or left_trim > 0 or right_trim < w:
        return cell_img[top_trim:bottom_trim, left_trim:right_trim]
    
    return cell_img


# ─────────────────────────────────────────────
#  CROP DARI GRID — PRESISI (margin mm)
# ─────────────────────────────────────────────

def crop_cells_from_grid(image, gray, row_pos, col_pos,
                          h_thick, v_thick,
                          margin_px_h, margin_px_v):
    """
    Crop setiap sel dari grid dengan presisi tinggi dan rasio 1:1.

    Strategi:
    1. Margin Statis: offset dari posisi garis berdasarkan ketebalan garis
       terdeteksi (lebih stabil daripada smart-inset berbasis konten piksel
       yang mudah gagal pada variasi kualitas scan)
    2. Square Enforcement: paksa crop menjadi persegi (1:1) dengan center crop

    Hasilnya: crop persegi yang konsisten, tanpa garis batas, karakter utuh.
    """
    img_h, img_w = image.shape[:2]
    cells = []

    # Offset dari tepi garis: setengah ketebalan + margin user
    # Pastikan minimal 4px agar garis tidak ikut ter-crop
    off_h = max(margin_px_h, h_thick // 2 + 4)
    off_v = max(margin_px_v, v_thick // 2 + 4)

    for i in range(len(row_pos) - 1):
        row_cells = []
        for j in range(len(col_pos) - 1):
            # Batas sel dengan margin statis dari posisi garis
            y1 = row_pos[i]   + off_h
            y2 = row_pos[i+1] - off_h
            x1 = col_pos[j]   + off_v
            x2 = col_pos[j+1] - off_v

            # Clamp ke batas gambar
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(y1 + 1, min(y2, img_h))
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(x1 + 1, min(x2, img_w))

            cell_w = x2 - x1
            cell_h = y2 - y1

            # Abaikan sel terlalu kecil
            if cell_w < 15 or cell_h < 15:
                continue

            # ── Enforce Square (rasio 1:1) ──
            # Ambil sisi terpendek, lalu center crop
            side = min(cell_w, cell_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            sq_x1 = cx - side // 2
            sq_y1 = cy - side // 2
            sq_x2 = sq_x1 + side
            sq_y2 = sq_y1 + side

            # Geser jika keluar batas gambar
            if sq_x1 < 0:
                sq_x2 += abs(sq_x1); sq_x1 = 0
            if sq_y1 < 0:
                sq_y2 += abs(sq_y1); sq_y1 = 0
            if sq_x2 > img_w:
                shift = sq_x2 - img_w
                sq_x1 = max(0, sq_x1 - shift); sq_x2 = img_w
            if sq_y2 > img_h:
                shift = sq_y2 - img_h
                sq_y1 = max(0, sq_y1 - shift); sq_y2 = img_h

            cell = image[sq_y1:sq_y2, sq_x1:sq_x2]
            if cell.size > 0:
                row_cells.append({
                    'image': cell,
                    'row':   i,
                    'col':   j,
                    'bbox':  (sq_x1, sq_y1, sq_x2, sq_y2)
                })

        if row_cells:
            cells.append(row_cells)

    return cells


# ─────────────────────────────────────────────
#  METODE TEMPLATE (Bands & Local Columns)
# ─────────────────────────────────────────────

def _find_positions_template(profile, threshold, merge_dist=10):
    idx = np.where(profile >= threshold)[0]
    if len(idx) == 0:
        return []
    groups, current = [], [idx[0]]
    for i in idx[1:]:
        if i - current[-1] <= merge_dist:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return [int(np.mean(g)) for g in groups]

def detect_row_lines_template(binary, min_frac=0.4, dilate_px=5, merge_dist=15):
    w = binary.shape[1]
    dilated = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, dilate_px)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horiz_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    
    # Tambahkan dilatasi vertikal agar garis yang miring menumpuk di koordinat y yang sama
    vert_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    horiz_mask = cv2.dilate(horiz_mask, vert_dilate, iterations=1)
    
    profile = (horiz_mask > 0).sum(axis=1)
    return sorted(_find_positions_template(profile, w * 0.2, merge_dist)), horiz_mask

def classify_response_bands(row_lines):
    heights = np.diff(row_lines)
    if len(heights) <= 1:
        return np.array([True] * len(heights))
    order = np.argsort(heights)
    sorted_h = heights[order]
    gaps = np.diff(sorted_h)
    split_idx = np.argmax(gaps)
    threshold = (sorted_h[split_idx] + sorted_h[split_idx + 1]) / 2
    return heights > threshold

def detect_col_lines_in_band(vert_mask, y0, y1, min_frac=0.4, merge_dist=10):
    band = vert_mask[y0:y1, :]
    profile = (band > 0).sum(axis=0)
    threshold = (y1 - y0) * min_frac
    return sorted(_find_positions_template(profile, threshold, merge_dist))

def build_cells_template(image, expected_cols=3, min_frac_row=0.4, min_frac_col=0.4, dilate_px=5, response_rows_only=True, margin_px=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape

    row_lines, _ = detect_row_lines_template(binary, min_frac_row, dilate_px)
    if len(row_lines) < 2:
        return []

    row_lines_arr = np.array(row_lines)
    is_response = classify_response_bands(row_lines_arr)

    dilated_h = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, 1)))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 200, 5)))
    vert_mask = cv2.morphologyEx(dilated_h, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    cells = []
    skipped = 0
    row_idx = 0
    for i in range(len(row_lines) - 1):
        y0, y1 = row_lines[i], row_lines[i + 1]

        if response_rows_only and not is_response[i]:
            skipped += 1
            continue

        cols = detect_col_lines_in_band(vert_mask, y0, y1, min_frac_col)

        if len(cols) - 1 != expected_cols:
            left = min(cols) if cols else 0
            right = max(cols) if cols else w
            cols = list(np.linspace(left, right, expected_cols + 1).astype(int))

        row_cells = []
        for j in range(len(cols) - 1):
            x0, x1 = cols[j], cols[j + 1]
            # Tambahkan margin kecil agar crop tidak persis mengenai garis
            cell_margin = margin_px
            cy0 = max(0, y0 + cell_margin)
            cy1 = min(h, y1 - cell_margin)
            cx0 = max(0, x0 + cell_margin)
            cx1 = min(w, x1 - cell_margin)
            
            if (cy1 - cy0) > 10 and (cx1 - cx0) > 10:
                cell_img = image[cy0:cy1, cx0:cx1]
                row_cells.append({
                    'image': cell_img,
                    'row': row_idx,
                    'col': j,
                    'bbox': (cx0, cy0, cx1, cy1)
                })
        
        if row_cells:
            cells.append(row_cells)
            row_idx += 1

    print(f"  [Template] -> {len(row_lines)-1} baris terdeteksi, {skipped} baris contoh dilewati.")
    return cells

# ─────────────────────────────────────────────
#  METODE KONTUR (fallback terakhir)
# ─────────────────────────────────────────────

def find_contour_boxes(image, min_area=5000, min_aspect=0.3, max_aspect=5.0):
    """Deteksi kotak individual menggunakan kontur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if min_aspect <= aspect <= max_aspect:
            boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: (b[1] // 50, b[0]))
    return boxes


def crop_with_contours(image, boxes, margin_px=10):
    """Crop berdasarkan bounding box kontur dengan margin."""
    img_h, img_w = image.shape[:2]
    cropped = []
    for idx, (x, y, bw, bh) in enumerate(boxes):
        x1 = max(0, x + margin_px)
        y1 = max(0, y + margin_px)
        x2 = min(img_w, x + bw - margin_px)
        y2 = min(img_h, y + bh - margin_px)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            cropped.append({
                'image': image[y1:y2, x1:x2],
                'index': idx,
                'bbox':  (x1, y1, x2, y2)
            })
    return cropped


# ─────────────────────────────────────────────
#  HELPER DETEKSI PERSEGI & WARNA MERAH
# ─────────────────────────────────────────────

def is_square_cell(bbox, img_shape, min_dim_ratio=0.06):
    """
    Memeriksa apakah sebuah sel memiliki bentuk mendekati persegi (square)
    dan berukuran cukup besar untuk menyaring garis label/metadata/noise.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    # Rasio aspek mendekati 1.0 (antara 0.82 dan 1.20)
    is_aspect_ok = 0.55 <= aspect <= 1.85
    
    # Batas ukuran minimum (persentase dari ukuran gambar)
    img_h, img_w = img_shape[:2]
    min_size = min(img_h, img_w) * min_dim_ratio
    is_size_ok = (w >= min_size) and (h >= min_size)
    
    return is_aspect_ok and is_size_ok


def analyze_red_color(cell_img, threshold_ratio=0.0003):
    """
    Mendeteksi keberadaan warna merah di dalam sel (misal coretan/centang).
    Mengembalikan: (is_red_detected, cleaned_image)
    Di mana cleaned_image adalah gambar yang coretan merahnya telah dibersihkan (diubah ke putih).
    """
    if cell_img is None or cell_img.size == 0:
        return False, cell_img
        
    # Pastikan gambar dalam format BGR (3 channel)
    if len(cell_img.shape) != 3:
        return False, cell_img
        
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    
    # Range warna merah di HSV (dua bagian)
    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 40])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    red_pixel_count = np.sum(red_mask > 0)
    total_pixels = cell_img.shape[0] * cell_img.shape[1]
    
    is_red = red_pixel_count > (total_pixels * threshold_ratio)
    
    # Bersihkan coretan merah dengan mengubahnya menjadi putih
    cleaned = cell_img.copy()
    if is_red:
        # Dilasi mask merah sedikit agar tepian warna merah ikut bersih
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_mask = cv2.dilate(red_mask, kernel, iterations=1)
        cleaned[dilated_mask > 0] = [255, 255, 255]
        
    return is_red, cleaned


def has_writing(cell_img, min_ink_ratio=0.0015):
    """
    Mendeteksi apakah sebuah sel memiliki tulisan aksara di dalamnya (tidak kosong).
    Menggunakan pemeriksaan kontras untuk menghindari pembesaran noise Otsu pada kertas kosong,
    serta mendeteksi goresan tipis dengan aman.
    """
    if cell_img is None or cell_img.size == 0:
        return False
        
    # Konversi ke grayscale
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    
    # Ambil area tengah (70% tengah)
    h, w = gray.shape[:2]
    by = int(h * 0.15)
    bx = int(w * 0.15)
    center_area = gray[by:h-by, bx:w-bx]
    
    if center_area.size == 0:
        return False
        
    # Periksa kontras (nilai maksimum - minimum) untuk mendeteksi kertas kosong secara absolut
    min_val, max_val, _, _ = cv2.minMaxLoc(center_area)
    contrast = max_val - min_val
    if contrast < 30:  # Kertas putih polos biasanya memiliki kontras < 20
        return False
        
    # Binarisasi: tinta hitam jadi putih (255)
    _, thresh = cv2.threshold(center_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    ink_pixels = np.sum(thresh > 0)
    total_pixels = center_area.size
    ink_ratio = ink_pixels / total_pixels
    
    return ink_ratio >= min_ink_ratio


# ─────────────────────────────────────────────
#  RESIZE & SIMPAN
# ─────────────────────────────────────────────

def resize_cell(cell_img, target_size=(128, 128)):
    """
    Sesuai instruksi: Crop gambar dari tengah menjadi kotak persegi (center square crop),
    kemudian resize ke target_size. Tidak ada padding atau mirror yang ditambahkan.
    """
    h, w = cell_img.shape[:2]
    if h == 0 or w == 0:
        return np.ones((target_size[0], target_size[1]), dtype=np.uint8) * 255

    # Ambil ukuran sisi terkecil untuk membuat kotak sempurna (Square)
    size = min(h, w)
    
    # Hitung koordinat tengah
    y1 = (h - size) // 2
    y2 = y1 + size
    x1 = (w - size) // 2
    x2 = x1 + size
    
    # Potong (Crop) jadi kotak sempurna dari tengah
    square_img = cell_img[y1:y2, x1:x2]
    
    # Resize ke target (128x128)
    return cv2.resize(square_img, target_size, interpolation=cv2.INTER_AREA)

def get_box_letter(idx):
    """
    Konversi indeks 0-based ke urutan huruf:
    0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', 27 -> 'ab', dst.
    """
    res = []
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        res.append(chr(97 + rem))
    return ''.join(reversed(res))


def save_results(cells_flat, output_dir, stem, resize_output=True, target_size=(128, 128), group_by_box=False, out_dpi=100, ext='.jpg'):
    """Simpan setiap sel crop ke file gambar sesuai format asli dengan penamaan _a, _b, dst."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    if not ext.startswith('.'):
        ext = '.' + ext
    ext = ext.lower()

    # Jika group_by_box aktif, petakan row & col ke nomor Box (1 s.d. N)
    box_mapping = {}
    if group_by_box:
        rows_list = sorted(list(set(cell['row'] for cell in cells_flat if 'row' in cell)))
        cols_list = sorted(list(set(cell['col'] for cell in cells_flat if 'col' in cell)))
        for cell in cells_flat:
            if 'row' in cell and 'col' in cell:
                r_idx = rows_list.index(cell['row'])
                c_idx = cols_list.index(cell['col'])
                # Box number = r_idx * jumlah_kolom + c_idx + 1
                box_num = r_idx * len(cols_list) + c_idx + 1
                box_mapping[id(cell)] = f"Box {box_num}"
            else:
                # Fallback untuk metode kontur
                box_mapping[id(cell)] = f"Box {cell.get('index', 0) + 1}"

    for idx, cell in enumerate(cells_flat):
        img = cell['image']
        if img is None or img.size == 0:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        out  = resize_cell(gray, target_size) if resize_output else gray

        letter = get_box_letter(idx)
        fname = f"{stem}_{letter}{ext}"

        if group_by_box:
            box_folder_name = box_mapping.get(id(cell), "Box Lain")
            dest_dir = os.path.join(output_dir, box_folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            path = os.path.join(dest_dir, fname)
        else:
            path = os.path.join(output_dir, fname)

        if out_dpi and out_dpi > 0:
            try:
                from PIL import Image
                if len(out.shape) == 2:
                    img_pil = Image.fromarray(out)
                else:
                    img_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                if ext in ('.jpg', '.jpeg'):
                    img_pil.save(path, dpi=(out_dpi, out_dpi), quality=95)
                else:
                    img_pil.save(path, dpi=(out_dpi, out_dpi))
            except Exception as e:
                print(f"[WARN] Gagal set DPI dengan Pillow, fallback ke cv2: {e}")
                cv2.imwrite(path, out)
        else:
            cv2.imwrite(path, out)
            
        saved.append(path)

    return saved


# ─────────────────────────────────────────────
#  DEBUG VISUALIZATION
# ─────────────────────────────────────────────

def draw_debug_view(image, cells_flat, h_thick=3, v_thick=3,
                    row_pos=None, col_pos=None):
    """Visualisasi debug: kotak crop + garis grid."""
    debug = image.copy()

    # Gambar garis grid biru tipis
    h_img, w_img = debug.shape[:2]
    if row_pos:
        for y in row_pos:
            cv2.line(debug, (0, y), (w_img, y), (255, 100, 0), 1)
    if col_pos:
        for x in col_pos:
            cv2.line(debug, (x, 0), (x, h_img), (255, 100, 0), 1)

    # Gambar kotak crop hijau dengan label huruf _a, _b, ...
    for idx, cell in enumerate(cells_flat):
        bbox = cell.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 80), 2)
            letter = get_box_letter(idx)
            label = f"_{letter}"
            cv2.putText(debug, label, (x1 + 3, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)

    # Info header
    n = len(cells_flat)
    info = f"{n} sel | LineH:{h_thick}px V:{v_thick}px"
    cv2.rectangle(debug, (0, 0), (280, 28), (20, 20, 20), -1)
    cv2.putText(debug, info, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 130), 1)
    return debug


# ─────────────────────────────────────────────
#  CROP FOKUS KARAKTER
# ─────────────────────────────────────────────

def crop_to_content_box(cell_img, border_margin_ratio=0.08, pad_ratio=0.12):
    """
    Buang margin dekat garis tabel, lalu crop tight ke bounding box tinta + padding proporsional.
    Berdasarkan logika referensi pengguna untuk menghilangkan sisa garis tabel.
    """
    if cell_img is None or cell_img.size == 0:
        return cell_img

    # Konversi ke grayscale jika berwarna
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    
    h, w = gray.shape
    m_y, m_x = int(h * border_margin_ratio), int(w * border_margin_ratio)
    inner = gray[m_y:h - m_y, m_x:w - m_x]

    if inner.size == 0:
        return None

    _, binary = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None  # sel kosong, tidak ada tinta

    x, y, w2, h2 = cv2.boundingRect(coords)
    
    # Translasi koordinat inner ke koordinat gambar asli (gray)
    x += m_x
    y += m_y
    
    cx = x + w2 / 2.0
    cy = y + h2 / 2.0
    
    # Gunakan sisi terpanjang agar menjadi persegi
    side = max(w2, h2)
    pad = int(side * pad_ratio)
    half_side = int((side / 2) + pad)
    
    x0 = int(cx - half_side)
    y0 = int(cy - half_side)
    x1 = int(cx + half_side)
    y1 = int(cy + half_side)
    
    # Geser crop box jika keluar batas gambar asli agar tetap sebisa mungkin persegi
    if x0 < 0:
        x1 += abs(x0); x0 = 0
    if y0 < 0:
        y1 += abs(y0); y0 = 0
    if x1 > w:
        x0 -= (x1 - w); x1 = w
    if y1 > h:
        y0 -= (y1 - h); y1 = h
        
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)

    return gray[y0:y1, x0:x1]

def process_image(image_path, output_dir,
                  method='auto',
                  margin_mm=1.0,
                  resize_output=True,
                  target_size=(128, 128),
                  min_area=5000,
                  save_debug=True,
                  only_squares=False,
                  only_red=False,
                  clean_red=False,
                  exclude_top_pct=0.28,
                  exclude_bottom_pct=0.05,
                  exclude_empty=True,
                  group_by_box=False,
                  padding=None,          # backward compat
                  deskew=True,
                  crop_to_content=True,
                  manual_boxes=None,
                  out_dpi=100,
                  return_boxes_only=False,
                  exact_boxes=None,
                  original_filename=None,
                  ext=None):
    """
    Proses satu gambar: deteksi grid/box → crop → simpan.

    margin_mm    : Jarak dari tepi garis (dalam mm). Default 1mm.
                   Garis tidak akan ikut dalam hasil crop.
    only_squares : Hanya simpan box yang berbentuk persegi (membuang label metadata).
    only_red     : Hanya simpan box yang terdeteksi ada coretan/centang warna merah.
    clean_red    : Bersihkan/hapus warna merah dari hasil crop akhir.
    exclude_top_pct    : Persentase tinggi gambar dari atas yang dibuang (header/metadata). Default 0.28 (28%).
    exclude_bottom_pct : Persentase tinggi gambar dari bawah yang dibuang (footer/noise). Default 0.05 (5%).
    exclude_empty      : Buang box kosong yang tidak ada aksara/tulisan tangan di dalamnya.
    group_by_box       : Jika True, simpan dalam folder Box 1, Box 2 dst. Bukan per nama gambar.
    """
    import re
    from datetime import datetime

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Tidak dapat membuka: {image_path}")

    # Luruskan kemiringan gambar secara otomatis jika miring (scan/foto)
    if deskew:
        image, skew_angle = deskew_image(image)
        # Transformasi manual_boxes agar posisinya ikut berputar
        if abs(skew_angle) > 0.05 and manual_boxes:
            import math
            h, w = image.shape[:2]
            cx_img = w / 2.0
            cy_img = h / 2.0
            rad = math.radians(skew_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            for b in manual_boxes:
                bx = b['x'] * w
                by = b['y'] * h
                bw = b['w'] * w
                bh = b['h'] * h
                bcx = bx + bw/2.0
                bcy = by + bh/2.0
                # Translasi ke origin pusat gambar
                tx = bcx - cx_img
                ty = bcy - cy_img
                # Putar (kebalikan rotasi gambar)
                rx = tx * cos_a + ty * sin_a
                ry = -tx * sin_a + ty * cos_a
                new_bcx = rx + cx_img
                new_bcy = ry + cy_img
                b['x'] = (new_bcx - bw/2.0) / w
                b['y'] = (new_bcy - bh/2.0) / h

        if abs(skew_angle) > 0.05 and exact_boxes:
            import math
            h, w = image.shape[:2]
            cx_img = w / 2.0
            cy_img = h / 2.0
            rad = math.radians(skew_angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            for b in exact_boxes:
                bx = b['x'] * w
                by = b['y'] * h
                bw = b['w'] * w
                bh = b['h'] * h
                bcx = bx + bw/2.0
                bcy = by + bh/2.0
                tx = bcx - cx_img
                ty = bcy - cy_img
                rx = tx * cos_a + ty * sin_a
                ry = -tx * sin_a + ty * cos_a
                new_bcx = rx + cx_img
                new_bcy = ry + cy_img
                b['x'] = (new_bcx - bw/2.0) / w
                b['y'] = (new_bcy - bh/2.0) / h
    else:
        skew_angle = 0.0

    if original_filename:
        orig_p = Path(original_filename)
        stem = orig_p.stem
        if ext is None:
            ext = orig_p.suffix if orig_p.suffix else Path(image_path).suffix
    else:
        orig_p = Path(image_path)
        raw_stem = orig_p.stem
        stem = re.sub(r'^[0-9a-fA-F]{10}_', '', raw_stem)
        if ext is None:
            ext = orig_p.suffix

    if not ext:
        ext = '.jpg'
    if not ext.startswith('.'):
        ext = '.' + ext
    ext = ext.lower()

    sub_output = os.path.join(output_dir, stem)
    if not group_by_box:
        os.makedirs(sub_output, exist_ok=True)

    gray, thresh = preprocess_image(image)

    dpi = estimate_dpi(image.shape)
    
    # Margin dinamis: 2.5mm untuk lurus, 6.0mm untuk miring (seperti request user)
    actual_margin = 6.0 if abs(skew_angle) > 0.5 else 2.5
    print(f"[INFO] Margin disesuaikan dinamis ke {actual_margin}mm (Skew: {skew_angle:.2f}°)")

    if padding is not None:
        margin_px_h = margin_px_v = max(int(padding), 4)
        print(f"[INFO] Padding: {margin_px_h}px (DPI={dpi:.0f})")
    else:
        margin_px_h = max(mm_to_px(actual_margin, dpi), 4)  # Minimum 4px
        margin_px_v = margin_px_h
        print(f"[INFO] Margin: {actual_margin}mm = {margin_px_h}px (DPI={dpi:.0f})")

    cells_flat = []
    row_pos = col_pos = None
    h_thick = v_thick = 3

    # ── METODE EXACT (Langsung potong sesuai list koordinat box, bypass extrapolasi) ──
    if exact_boxes:
        h, w = image.shape[:2]
        for idx, box in enumerate(exact_boxes):
            fx1 = max(0, int(box['x'] * w))
            fy1 = max(0, int(box['y'] * h))
            fx2 = min(w, int((box['x'] + box['w']) * w))
            fy2 = min(h, int((box['y'] + box['h']) * h))
            
            cells_flat.append({
                'image': image[fy1:fy2, fx1:fx2],
                'row': 0,
                'col': idx,
                'bbox': (fx1, fy1, fx2, fy2)
            })

    # ── METODE MANUAL ──────────────────────────
    elif method == 'manual' and manual_boxes:
        print(f"[MANUAL] Memotong berdasarkan {len(manual_boxes)} kotak penanda manual pada: {stem}")
        h, w = image.shape[:2]
        debug_img = image.copy()

        # ── DETEKSI GARIS FISIK TABEL (H + V via HoughLinesP) ─────────────
        edges = cv2.Canny(gray, 30, 100)

        def run_hough(edges, min_len, gap=25, thr=40):
            return cv2.HoughLinesP(edges, 1, np.pi/180,
                                   threshold=thr, minLineLength=min_len, maxLineGap=gap)

        def cluster_positions(raw, gap=12):
            if not raw: return []
            raw = sorted(set(raw))
            clusters, cur = [], [raw[0]]
            for v in raw[1:]:
                if v - cur[-1] <= gap: cur.append(v)
                else:
                    clusters.append(int(np.mean(cur))); cur = [v]
            clusters.append(int(np.mean(cur)))
            return clusters

        # ── Garis HORIZONTAL: dual-pass agar menangkap semua garis ──────────
        # Pass 1: garis pendek (10% lebar), gap besar untuk garis putus-putus
        # Pass 2: garis panjang (25% lebar), threshold lebih tinggi
        raw_h_pos, h_extents = [], []
        for min_h_pct, thr_h, gap_h in [(0.10, 30, 60), (0.25, 50, 40)]:
            min_h_len = max(25, int(w * min_h_pct))
            raw_h = run_hough(edges, min_h_len, gap=gap_h, thr=thr_h)
            if raw_h is not None:
                for l in raw_h:
                    x1, y1, x2, y2 = l[0]
                    ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                    if ang < 8 or ang > 172:
                        ym = (y1+y2)//2
                        raw_h_pos.append(ym)
                        h_extents.append((min(x1,x2), max(x1,x2), ym))

        # Cluster H lines
        h_lines_raw = sorted([y for y in cluster_positions(raw_h_pos, gap=12)
                               if 3 < y < h-3])

        # ── FILTER H lines berdasarkan jangkauan horizontal (coverage) ────────
        # Garis grid fisik (walau putus-putus) bersama-sama menjangkau lebar
        # yang besar (leftmost_start s.d. rightmost_end ≥ 20% lebar gambar).
        # Stroke aksara tulisan tangan = segmen tunggal pendek → range kecil.
        MIN_COVERAGE_PCT = 0.20  # jangkauan minimal 20% lebar gambar
        h_lines_all = []
        for y in h_lines_raw:
            # Kumpulkan semua extent segmen yang dekat dengan y (dalam ±15px)
            nearby = [(xs, xe) for xs, xe, ey in h_extents if abs(ey - y) <= 15]
            if nearby:
                leftmost  = min(xs for xs, xe in nearby)
                rightmost = max(xe for xs, xe in nearby)
                coverage  = rightmost - leftmost  # total range yang dijangkau
                if coverage >= w * MIN_COVERAGE_PCT:
                    h_lines_all.append(y)
        # Jika filter terlalu agresif (< 2 H lines), gunakan semua
        if len(h_lines_all) < 2:
            h_lines_all = h_lines_raw

        # ── Garis VERTIKAL: Hough dengan vote filter adaptif ─────────────
        min_v_len = max(30, int(h * 0.10))
        raw_v = run_hough(edges, min_v_len, gap=15, thr=35)
        from collections import Counter
        v_votes = Counter()
        if raw_v is not None:
            for l in raw_v:
                x1, y1, x2, y2 = l[0]
                ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if 82 < ang < 98:
                    v_votes[((x1+x2)//2 // 8) * 8] += 1

        # Mulai dengan vote ≥2; jika kurang dari 3 garis, turunkan ke ≥1
        raw_v_pos = [x for x, cnt in v_votes.items() if cnt >= 2]
        if len(raw_v_pos) < 3:
            raw_v_pos = [x for x, cnt in v_votes.items() if cnt >= 1]

        # Tambahkan batas kiri/kanan tabel dari H lines
        # Gunakan semua H extents yang panjang (≥15% lebar gambar)
        long_h  = [(xs,xe,y) for xs,xe,y in h_extents if (xe-xs) > w*0.15]
        short_h = [(xs,xe,y) for xs,xe,y in h_extents]
        if long_h:
            raw_v_pos.append(int(np.percentile([xs for xs,xe,y in long_h], 5)))
            raw_v_pos.append(int(np.percentile([xe for xs,xe,y in long_h], 95)))
        if short_h:
            # Batas mutlak kiri/kanan dari SEMUA segmen H
            raw_v_pos.append(int(np.percentile([xs for xs,xe,y in short_h], 2)))
            raw_v_pos.append(int(np.percentile([xe for xs,xe,y in short_h], 98)))

        # Backup morfologi V (kernel lebih pendek agar lebih sensitif)
        dil = cv2.dilate(thresh, np.ones((3,1), np.uint8), iterations=1)
        vm = cv2.morphologyEx(dil,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h*0.10)))),
                              iterations=2)
        raw_v_pos += [int(x) for x in np.where(np.sum(vm>0, axis=0) >= int(h*0.05))[0]]

        v_lines_all = sorted([x for x in cluster_positions(raw_v_pos, gap=12)
                               if 3 < x < w-3])

        # ── INFERENSI V LINES DARI POSISI MARKER USER ─────────────────────
        if manual_boxes:
            first_row = sorted([b for b in manual_boxes], key=lambda b: b['x'])
            n_cols = max(len([b for b in manual_boxes if abs(b['y'] - manual_boxes[0]['y']) < 0.05]), 3)
            col_centers_px = sorted(set([int((b['x'] + b['w']/2) * w) for b in manual_boxes]))

            if len(v_lines_all) < n_cols + 1 and len(col_centers_px) >= 2:
                col_gaps = [col_centers_px[i+1] - col_centers_px[i]
                            for i in range(len(col_centers_px)-1)]
                avg_col_w = int(np.mean(col_gaps)) if col_gaps else 190

                tbl_left  = min(v_lines_all) if v_lines_all else col_centers_px[0]  - avg_col_w//2
                tbl_right = max(v_lines_all) if v_lines_all else col_centers_px[-1] + avg_col_w//2

                for cx in col_centers_px:
                    v_lines_all.append(cx - avg_col_w // 2)
                    v_lines_all.append(cx + avg_col_w // 2)

                v_lines_all.append(tbl_left)
                v_lines_all.append(tbl_right)

                v_lines_all = sorted(set([x for x in cluster_positions(v_lines_all, gap=12)
                                         if 3 < x < w-3]))

        print(f"[MANUAL] H lines ({len(h_lines_all)}): {h_lines_all}")
        print(f"[MANUAL] V lines ({len(v_lines_all)}): {v_lines_all}")

        # ── TENTUKAN PERGESERAN GLOBAL (GLOBAL SHIFT) ─────────────
        def get_nearest(lines, target, max_dist=40):
            valid = [l for l in lines if abs(l - target) <= max_dist]
            return min(valid, key=lambda l: abs(l - target)) if valid else target

        dx_votes = []
        dy_votes = []
        cell_w_votes = []
        cell_h_votes = []
        for b in manual_boxes:
            bx = int(b['x'] * w)
            by = int(b['y'] * h)
            bw = int(b['w'] * w)
            bh = int(b['h'] * h)
            
            r_l = get_nearest(v_lines_all, bx, max_dist=80)
            r_r = get_nearest(v_lines_all, bx + bw, max_dist=80)
            if r_l != bx and r_r != (bx + bw):
                dx = ((r_l + r_r) / 2.0) - (bx + bw / 2.0)
                dx_votes.append(dx)
                cell_w_votes.append(r_r - r_l)
            
            r_t = get_nearest(h_lines_all, by, max_dist=80)
            r_b = get_nearest(h_lines_all, by + bh, max_dist=80)
            if r_t != by and r_b != (by + bh):
                dy = ((r_t + r_b) / 2.0) - (by + bh / 2.0)
                dy_votes.append(dy)
                cell_h_votes.append(r_b - r_t)

        # Median shift untuk mengabaikan noise (misal satu kotak salah deteksi garis karena tulisan)
        global_dx = np.median(dx_votes) if dx_votes else 0.0
        global_dy = np.median(dy_votes) if dy_votes else 0.0
        print(f"[MANUAL] Global shift dari raw lines: dx={global_dx:.1f}, dy={global_dy:.1f}")

        # ── KELOMPOKKAN KE DALAM BARIS DAN KOLOM MATEMATIS ─────────────
        manual_boxes.sort(key=lambda b: b['y'])
        rows_group = []
        current_row = []
        for b in manual_boxes:
            if not current_row:
                current_row.append(b)
            elif abs(b['y'] - current_row[-1]['y']) < 0.05:
                current_row.append(b)
            else:
                current_row.sort(key=lambda box: box['x'])
                rows_group.append(current_row)
                current_row = [b]
        if current_row:
            current_row.sort(key=lambda box: box['x'])
            rows_group.append(current_row)

        # Hitung titik tengah IDEAL dari template (memaksa grid menjadi sempurna dan rata)
        row_y_centers = []
        for row in rows_group:
            row_y_centers.append(np.median([int(b['y']*h + b['h']*h/2.0) for b in row]))
            
        num_cols = max(len(row) for row in rows_group)
        col_x_centers = []
        for c in range(num_cols):
            col_boxes = [row[c] for row in rows_group if c < len(row)]
            if col_boxes:
                col_x_centers.append(np.median([int(b['x']*w + b['w']*w/2.0) for b in col_boxes]))

        # Ukuran sel global agar semuanya 100% identik ukurannya
        if cell_w_votes and cell_h_votes:
            global_square_size = int(min(np.median(cell_w_votes), np.median(cell_h_votes)))
        else:
            global_square_size = int(np.mean([min(b['w']*w, b['h']*h) for b in manual_boxes]))

        print(f"[MANUAL] Global Square Size: {global_square_size}px")

        box_counter = 1
        for r_idx, row_boxes in enumerate(rows_group):
            row_boxes.sort(key=lambda b: b['x'])
            for c_idx, box in enumerate(row_boxes):
                bx = int(box['x'] * w)
                by = int(box['y'] * h)
                bw = int(box['w'] * w)
                bh = int(box['h'] * h)
                # 1. Ambil titik tengah IDEAL matematis dari template
                ideal_cx = col_x_centers[c_idx] if c_idx < len(col_x_centers) else int(box['x']*w + box['w']*w/2.0)
                ideal_cy = row_y_centers[r_idx] if r_idx < len(row_y_centers) else int(box['y']*h + box['h']*h/2.0)
                
                # 2. Terapkan translasi global (menggeser seluruh grid sekaligus)
                pc_cx = ideal_cx + global_dx
                pc_cy = ideal_cy + global_dy
                
                # 3. Buat KOTAK PERSEGI SAMA SISI menggunakan global_square_size
                # Gunakan penambahan langsung untuk menghindari ukuran selisih 1 pixel akibat pembulatan
                fx1 = int(round(pc_cx - global_square_size / 2.0))
                fy1 = int(round(pc_cy - global_square_size / 2.0))
                fx2 = fx1 + global_square_size
                fy2 = fy1 + global_square_size
                
                disp_left = fx1
                disp_right = fx2
                disp_top = fy1
                disp_bottom = fy2
                
                # Terapkan penyusutan margin (perkecil kotak hijau)
                fx1 += margin_px_h
                fy1 += margin_px_v
                fx2 -= margin_px_h
                fy2 -= margin_px_v

                # Clamp ke batas gambar
                fx1 = max(0, min(fx1, w - 1))
                fy1 = max(0, min(fy1, h - 1))
                fx2 = max(fx1 + 1, min(fx2, w))
                fy2 = max(fy1 + 1, min(fy2, h))

                if fx2 <= fx1 or fy2 <= fy1:
                    print(f"[WARN] Kotak r{r_idx}c{c_idx} invalid setelah clamp, skip.")
                    box_counter += 1
                    continue

                cells_flat.append({
                    'image': image[fy1:fy2, fx1:fx2],
                    'row':   r_idx,
                    'col':   c_idx,
                    'bbox':  (fx1, fy1, fx2, fy2)
                })



                print(f"  [CROP] r{r_idx}c{c_idx}: marker=({int(bx)},{int(by)})-({int(bx+bw)},{int(by+bh)}) "
                      f"| crop=({fx1},{fy1})-({fx2},{fy2}) sisi={fx2-fx1}x{fy2-fy1}")

                # Debug: batas marker (biru) + crop (hijau)
                cv2.rectangle(debug_img, (disp_left, disp_top), (disp_right, disp_bottom), (255, 100, 0), 1)
                cv2.rectangle(debug_img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(debug_img, f"r{r_idx}c{c_idx}", (fx1+4, fy1+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
                box_counter += 1

    elif method == 'exact' or (exact_boxes and len(exact_boxes) > 0):
        h, w = image.shape[:2]
        debug_img = image.copy()
        row_pos, col_pos = [], []
        h_thick, v_thick = 2, 2
        
        for idx, box in enumerate(exact_boxes or []):
            bx = box['x'] * w
            by = box['y'] * h
            bw = box['w'] * w
            bh = box['h'] * h
            
            x1 = int(round(bx))
            y1 = int(round(by))
            x2 = int(round(bx + bw))
            y2 = int(round(by + bh))
            
            # Clamp ke batas gambar
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            r_idx = idx // 3
            c_idx = idx % 3
            
            cells_flat.append({
                'image': image[y1:y2, x1:x2],
                'row': r_idx,
                'col': c_idx,
                'bbox': (x1, y1, x2, y2)
            })
            
            letter = get_box_letter(idx)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"_{letter}", (x1 + 4, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    elif method in ('auto', 'grid'):
        rows, cols, h_thick, v_thick, horiz_m, vert_m = detect_grid(gray, thresh, image.shape)
        if len(rows) >= 2 and len(cols) >= 2:
            row_pos, col_pos = rows, cols
            cells_nested = crop_cells_from_grid(image, gray, rows, cols, h_thick, v_thick, margin_px_h, margin_px_v)
            cells_flat = [c for r in cells_nested for c in r]
    elif method == 'template':
        cells_nested = build_cells_template(image, margin_px=margin_px_h)
        cells_flat = [c for r in cells_nested for c in r]
    elif method == 'contour':
        boxes = find_contour_boxes(image, min_area=min_area)
        cells_flat = crop_with_contours(image, boxes, margin_px=margin_px_h)
    else:
        print(f"[WARN] Manual mode tidak ada boxes, atau method bukan manual: {stem}")
        return []

    if return_boxes_only:
        out_boxes = []
        h, w = image.shape[:2]
        import math
        for cell in cells_flat:
            x1, y1, x2, y2 = cell['bbox']
            
            if abs(skew_angle) > 0.05:
                # Reverse rotate
                cx_img = w / 2.0
                cy_img = h / 2.0
                # Rotate by -skew_angle
                rad = math.radians(-skew_angle)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                
                bcx = (x1 + x2) / 2.0
                bcy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1
                
                tx = bcx - cx_img
                ty = bcy - cy_img
                rx = tx * cos_a + ty * sin_a
                ry = -tx * sin_a + ty * cos_a
                new_bcx = rx + cx_img
                new_bcy = ry + cy_img
                
                out_boxes.append({
                    'x': (new_bcx - bw/2.0) / w,
                    'y': (new_bcy - bh/2.0) / h,
                    'w': bw / w,
                    'h': bh / h
                })
            else:
                out_boxes.append({
                    'x': x1 / w,
                    'y': y1 / h,
                    'w': (x2 - x1) / w,
                    'h': (y2 - y1) / h
                })
        return out_boxes

    # ── FILTERING & CLEANING WARNA MERAH / PERSEGI ──
    if method == 'exact':
        filtered_cells = cells_flat
    else:
        filtered_cells = []
        img_h = image.shape[0]
        for cell in cells_flat:
            x1, y1, x2, y2 = cell['bbox']

            # 0. Bersihkan sisa garis di tepi sel (akibat skew/kemiringan)
            cell['image'] = clean_border_lines(cell['image'])

            # 1. Filter Header (Metadata) & Footer (Noise bawah) berdasarkan posisi vertikal
            if method in ('auto', 'contour', 'grid'):
                if y1 < img_h * exclude_top_pct:
                    continue
                if y2 > img_h * (1.0 - exclude_bottom_pct):
                    continue

            # 2. Filter bentuk persegi jika diaktifkan (membuang label baris tipis)
            if only_squares and not is_square_cell(cell['bbox'], image.shape):
                continue

            # 3. Deteksi coretan/centang warna merah
            is_red, cleaned = analyze_red_color(cell['image'])
            cell['is_red'] = is_red

            # 4. Filter warna merah jika diaktifkan
            if only_red and not is_red:
                continue

            # 5. Bersihkan coretan merah jika diaktifkan
            img_to_check = cleaned if clean_red else cell['image']
            
            # 6. Potong ketat ke karakter isi (crop_to_content) jika diaktifkan
            if crop_to_content:
                cropped_content = crop_to_content_box(img_to_check)
                if cropped_content is None or cropped_content.size == 0:
                    if exclude_empty:
                        continue
                    # Jika tidak dibuang saat kosong, pertahankan gambar asli yang sudah dibersihkan
                else:
                    img_to_check = cropped_content

            # 7. Filter box kosong (yang tidak berisi aksara tulisan tangan)
            # Jika crop_to_content aktif, has_writing tetap dicek untuk memastikan tinta valid
            if exclude_empty and not has_writing(img_to_check):
                continue

            # Simpan hasil akhir ke objek sel
            cell['image'] = img_to_check

            filtered_cells.append(cell)

    cells_flat = filtered_cells
    
    # Normalisasi indeks baris (row) dan kolom (col) agar selalu dimulai dari 0
    # Ini memperbaiki penamaan file (r0c0 dst) meskipun baris header/contoh (row 0 fisik) telah difilter
    if cells_flat and 'row' in cells_flat[0]:
        min_row = min(c.get('row', 0) for c in cells_flat)
        min_col = min(c.get('col', 0) for c in cells_flat)
        for c in cells_flat:
            if 'row' in c: c['row'] -= min_row
            if 'col' in c: c['col'] -= min_col

    print(f"[FILTER] Tersisa {len(cells_flat)} sel setelah filter (only_squares={only_squares}, only_red={only_red})")

    if not cells_flat:
        print(f"[WARN] Tidak ada box yang lolos filter pada: {stem}")
        return []

    # ── SIMPAN ───────────────────────────────
    # Simpan ke folder Box X jika group_by_box=True, atau folder sub_output jika False
    save_dest_dir = output_dir if group_by_box else sub_output
    saved = save_results(cells_flat, save_dest_dir, stem, resize_output, target_size, group_by_box, out_dpi, ext=ext)
    print(f"[OK] {len(saved)} gambar tersimpan ke: {save_dest_dir}")

    if save_debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if method == 'exact' and 'debug_img' in locals() and debug_img is not None:
            debug = debug_img
        else:
            debug = draw_debug_view(image, cells_flat, h_thick, v_thick, row_pos, col_pos)
        # File utuh / debug disimpan di output_dir dengan timestamp di belakang nama file
        whole_img_path = os.path.join(output_dir, f"{stem}_{timestamp}{ext}")
        cv2.imwrite(whole_img_path, debug)
        print(f"[OK] File Utuh: {whole_img_path}")

    return saved


# ─────────────────────────────────────────────
#  BATCH
# ─────────────────────────────────────────────

def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def batch_process(input_dir, output_dir,
                  extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'),
                  **kwargs):
    """Proses semua gambar dalam satu folder."""
    all_images = [f for f in Path(input_dir).iterdir()
                  if f.suffix.lower() in extensions]
    if not all_images:
        print(f"[WARN] Tidak ada gambar di: {input_dir}")
        return
    all_images.sort(key=lambda p: natural_sort_key(p.name))
    print(f"[INFO] {len(all_images)} gambar ditemukan (terurut)")
    total = 0
    for p in all_images:
        try:
            saved  = process_image(str(p), output_dir, **kwargs)
            total += len(saved)
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
    print(f"\n[SELESAI] {total} gambar tersimpan ke: {output_dir}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Crop Otomatis Box pada Gambar (Tabel/Form)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python crop_boxes.py -i form.jpg -o hasil/
  python crop_boxes.py -i form.jpg -o hasil/ --margin-mm 1.5
  python crop_boxes.py -i folder/ -o hasil/ --batch
  python crop_boxes.py -i form.jpg -o hasil/ --no-resize --method grid
        """
    )
    p.add_argument('-i', '--input',     required=True)
    p.add_argument('-o', '--output',    default='output_crops')
    p.add_argument('--method',          choices=['auto', 'grid', 'contour', 'template'], default='auto')
    p.add_argument('--batch',           action='store_true')
    p.add_argument('--margin-mm',       type=float, default=1.0,
                   help='Margin dari tepi garis dalam mm (default: 1.0)')
    p.add_argument('--no-resize',       action='store_true')
    p.add_argument('--size',            type=int, nargs=2, default=[128, 128], metavar=('H', 'W'))
    p.add_argument('--min-area',        type=int, default=5000)
    p.add_argument('--no-debug',        action='store_true')
    p.add_argument('--no-deskew',       action='store_true', help='Nonaktifkan perataan kemiringan kertas otomatis')
    p.add_argument('--crop-to-content', action='store_true', help='Pangkas ketat ke karakter tulisan tangan di dalam kotak')
    args = p.parse_args()

    kwargs = dict(
        method=args.method,
        margin_mm=args.margin_mm,
        resize_output=not args.no_resize,
        target_size=tuple(args.size),
        min_area=args.min_area,
        save_debug=not args.no_debug,
        deskew=not args.no_deskew,
        crop_to_content=args.crop_to_content,
    )
    if args.batch:
        batch_process(args.input, args.output, **kwargs)
    else:
        process_image(args.input, args.output, **kwargs)


if __name__ == '__main__':
    main()
