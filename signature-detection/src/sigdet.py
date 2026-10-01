"""Inti pipeline deteksi keberadaan tanda tangan (classical image processing).

Alur: load -> luruskan orientasi -> crop ROI -> grayscale -> thresholding
      -> morphology -> ekstraksi fitur area -> aturan keputusan.
"""
from dataclasses import dataclass
import cv2
import numpy as np

# ROI relatif (x0, y0, x1, y1) terhadap citra yang sudah tegak (0..1).
# Relatif => tahan terhadap perbedaan resolusi (3506x2481 vs 1599x1132).
ROIS = {
    "rektor": (0.12, 0.715, 0.40, 0.822),   # tanda tangan Rektor (default)
    "dekan":  (0.50, 0.700, 0.80, 0.850),   # tanda tangan Dekan
}
# ROI "kosong" (hanya kertas) & "teks cetak" -> dipakai sbg sampel NEGATIF
NEG_ROIS = {
    "blank_kiri_atas":   (0.02, 0.02, 0.30, 0.15),
    "blank_kanan_atas":  (0.70, 0.02, 0.98, 0.15),
    "blank_kiri_tengah": (0.02, 0.30, 0.30, 0.43),
    "teks_cetak":        (0.05, 0.845, 0.33, 0.965),  # nama Rektor + nomor ijazah
}

# ---------- parameter aturan keputusan (dikalibrasi, lihat README) ----------
RULE = dict(
    min_fg_ratio=0.004,     # minimal foreground / luas ROI
    max_fg_ratio=0.20,      # di atas ini => ROI banjir foreground (noise / threshold salah)
    min_largest_cc=0.0025,  # komponen terhubung terbesar minimal (x luas ROI)
    min_cc_bbox_w=0.45,     # lebar bbox komponen terbesar minimal (x lebar ROI); teks cetak ~0.30, tanda tangan ~0.5-0.62
)


def load_upright(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    # Scan dokumen ini landscape tetapi tersimpan portrait (terputar 90 derajat)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def crop_rel(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].copy()


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ------------------------------ thresholding ------------------------------
def th_global(gray, t=110):
    """Global threshold tetap (ink gelap -> 255)."""
    _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return m


def th_otsu(gray):
    """Otsu: ambang dipilih otomatis dari histogram (setelah blur ringan)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    t, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m, float(t)


def th_adaptive(gray, block=None, C=12):
    """Adaptive Gaussian: ambang lokal per-jendela."""
    if block is None:
        block = int(round(gray.shape[1] * 0.06)) | 1   # ~6% lebar ROI, ganjil
        block = max(block, 15)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, C)


# ------------------------------ morphology --------------------------------
def morph_clean(mask, scale=1.0):
    """Opening (buang bintik noise) lalu closing (sambung goresan putus)."""
    k_open = max(2, int(round(3 * scale)))
    k_close = max(3, int(round(9 * scale)))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    return closed


def remove_border_lines(mask, border=0.02):
    """Hapus foreground kecil yang menempel di tepi ROI (potongan teks tetangga)."""
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    bx, by = int(w * border), int(h * border)
    for i in range(1, n):
        x, y, ww, hh, _ = stats[i]
        if (x <= bx and ww < w * 0.1) or (x + ww >= w - bx and ww < w * 0.1) \
           or (y <= by and hh < h * 0.15) or (y + hh >= h - by and hh < h * 0.15):
            out[lab == i] = 0
    return out


# ------------------------------ fitur & aturan ----------------------------
@dataclass
class Features:
    fg_pixels: int
    fg_ratio: float
    n_components: int
    largest_cc: float        # luas komponen terbesar / luas ROI
    largest_cc_w: float      # lebar bbox komponen terbesar / lebar ROI
    largest_cc_share: float  # luas komponen terbesar / total foreground


def extract_features(mask):
    h, w = mask.shape
    fg = int(np.count_nonzero(mask))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return Features(0, 0.0, 0, 0.0, 0.0, 0.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas))
    st = stats[i + 1]
    return Features(fg, fg / (h * w), n - 1, areas[i] / (h * w),
                    st[cv2.CC_STAT_WIDTH] / w, areas[i] / max(fg, 1))


def decide(f: Features, rule=RULE):
    ok = (rule["min_fg_ratio"] <= f.fg_ratio <= rule["max_fg_ratio"]
          and f.largest_cc >= rule["min_largest_cc"]
          and f.largest_cc_w >= rule["min_cc_bbox_w"])
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"


def run_method(gray, method, scale):
    """Kembalikan (mask_mentah, mask_bersih, info) untuk satu metode threshold."""
    if method == "global":
        raw, info = th_global(gray), "T=110"
    elif method == "otsu":
        raw, t = th_otsu(gray)
        info = f"T={t:.0f}"
    elif method == "adaptive":
        raw, info = th_adaptive(gray), "gauss"
    else:
        raise ValueError(method)
    clean = remove_border_lines(morph_clean(raw, scale))
    return raw, clean, info
