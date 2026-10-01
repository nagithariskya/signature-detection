"""Menjalankan seluruh eksperimen miniproject.

Contoh:
    python src/run_experiment.py --input data/samples --output results --roi rektor
    python src/run_experiment.py --image data/samples/01_HighQuality_Enhanced.jpg   # 1 citra saja
"""
import argparse, glob, os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sigdet as S

METHODS = ["global", "otsu", "adaptive"]


def erase_signature(roi_bgr, scale):
    """Buat sampel NEGATIF 'ROI tanpa tanda tangan': goresan dihapus via inpainting."""
    g = S.to_gray(roi_bgr)
    m = S.th_adaptive(g)
    k = max(5, int(15 * scale)) | 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.inpaint(roi_bgr, m, 5, cv2.INPAINT_TELEA)


def build_samples(img, roi_name):
    """Daftar (nama, label_sebenarnya, crop BGR) untuk satu citra."""
    scale = img.shape[1] / 3506.0
    pos = S.crop_rel(img, S.ROIS[roi_name])
    items = [(f"ROI_{roi_name}", True, pos),
             (f"{roi_name}_dihapus", False, erase_signature(pos, scale))]
    for name, roi in S.NEG_ROIS.items():
        items.append((name, False, S.crop_rel(img, roi)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/samples")
    ap.add_argument("--image", default=None, help="proses satu file saja")
    ap.add_argument("--output", default="results")
    ap.add_argument("--roi", default="rektor", choices=list(S.ROIS))
    a = ap.parse_args()

    files = [a.image] if a.image else sorted(
        f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(a.input, f"*.{ext}")))
    os.makedirs(os.path.join(a.output, "crops"), exist_ok=True)
    os.makedirs(os.path.join(a.output, "figures"), exist_ok=True)

    rows = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        img = S.load_upright(f)
        scale = img.shape[1] / 3506.0
        for name, truth, crop in build_samples(img, a.roi):
            if name.startswith("ROI_"):
                cv2.imwrite(os.path.join(a.output, "crops", f"{base}_{name}.png"), crop)
            gray = S.to_gray(crop)
            for m in METHODS:
                raw, clean, info = S.run_method(gray, m, scale)
                ft = S.extract_features(clean)
                pred = S.decide(ft)
                rows.append(dict(image=base, sample=name, truth="PRESENT" if truth else "ABSENT",
                                 method=m, thr=info, fg_pixels=ft.fg_pixels, fg_ratio=round(ft.fg_ratio, 4),
                                 n_comp=ft.n_components, largest_cc=round(ft.largest_cc, 4),
                                 largest_cc_w=round(ft.largest_cc_w, 3),
                                 decision=pred.replace("SIGNATURE ", ""),
                                 correct=(pred.endswith("PRESENT") == truth)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.output, "results.csv"), index=False)

    # ---- ringkasan akurasi per metode ----
    summ = []
    for m in METHODS:
        d = df[df.method == m]
        tp = ((d.truth == "PRESENT") & (d.decision == "PRESENT")).sum()
        fn = ((d.truth == "PRESENT") & (d.decision == "ABSENT")).sum()
        tn = ((d.truth == "ABSENT") & (d.decision == "ABSENT")).sum()
        fp = ((d.truth == "ABSENT") & (d.decision == "PRESENT")).sum()
        summ.append(dict(method=m, TP=tp, FN=fn, TN=tn, FP=fp, accuracy=round((tp + tn) / len(d), 3)))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(a.output, "summary.csv"), index=False)
    print("\n=== RINGKASAN ===\n", sdf.to_string(index=False))
    print("\n=== Kesalahan ===\n", df[~df.correct][["image", "sample", "method", "decision", "fg_ratio"]].to_string(index=False))

    # ---- figure 1: perbandingan metode + morphology (satu citra per kondisi) ----
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in ("01_HighQuality_Enhanced", "02_LowContrast", "04_HighNoise", "06_Faded_Underexposed") and a.image is None:
            continue
        img = S.load_upright(f)
        scale = img.shape[1] / 3506.0
        crop = S.crop_rel(img, S.ROIS[a.roi])
        gray = S.to_gray(crop)
        fig, ax = plt.subplots(3, 3, figsize=(13, 7))
        for r, m in enumerate(METHODS):
            raw, clean, info = S.run_method(gray, m, scale)
            ft = S.extract_features(clean)
            ax[r, 0].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"Grayscale ({m}, {info})", fontsize=9)
            ax[r, 1].imshow(raw, cmap="gray", vmin=0, vmax=255); ax[r, 1].set_title(f"Threshold mentah (fg={np.count_nonzero(raw)})", fontsize=9)
            ax[r, 2].imshow(clean, cmap="gray", vmin=0, vmax=255)
            ax[r, 2].set_title(f"+ opening/closing (fg={ft.fg_pixels}, {S.decide(ft)[10:]})", fontsize=9)
            for c in range(3): ax[r, c].axis("off")
        fig.suptitle(f"Perbandingan thresholding - {base}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.output, "figures", f"compare_{base}.png"), dpi=110)
        plt.close(fig)

    # ---- figure 2: sensitivitas threshold global (terlalu rendah / terlalu tinggi) ----
    d = os.path.dirname(files[0])
    refs = [("01_HighQuality_Enhanced", "kertas terang (median ~247)"),
            ("06_Faded_Underexposed", "kertas redup (median ~158)")]
    refs = [(r, t) for r, t in refs if os.path.exists(os.path.join(d, r + ".jpg"))]
    if refs:
        Ts = [40, 80, 110, 160, 200, 235]
        fig, ax = plt.subplots(len(refs), len(Ts), figsize=(16, 2.6 * len(refs) + 0.6))
        ax = np.atleast_2d(ax)
        plt.figure(figsize=(6.5, 3.8))
        for r, (name, desc) in enumerate(refs):
            img = S.load_upright(os.path.join(d, name + ".jpg"))
            gray = S.to_gray(S.crop_rel(img, S.ROIS[a.roi]))
            for i, t in enumerate(Ts):
                m = S.th_global(gray, t)
                ax[r, i].imshow(m, cmap="gray", vmin=0, vmax=255); ax[r, i].axis("off")
                ax[r, i].set_title(f"{name[:2]} | T={t} | fg={100*np.count_nonzero(m)/m.size:.1f}%", fontsize=9)
            Tr = np.arange(0, 256, 5)
            plt.plot(Tr, [np.count_nonzero(S.th_global(gray, t)) / gray.size for t in Tr], label=f"{name[:2]} {desc}")
        plt.figure(1)
        fig.suptitle("Sensitivitas global threshold pada ROI tanda tangan (putih = foreground)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(a.output, "figures", "threshold_sensitivity.png"), dpi=110)
        plt.close(fig)
        plt.figure(2)
        plt.xlabel("threshold T"); plt.ylabel("foreground ratio"); plt.yscale("log")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(a.output, "figures", "fg_ratio_vs_T.png"), dpi=110); plt.close("all")

    # ---- figure 3: overlay hasil keputusan pada citra penuh ----
    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    for axx, f in zip(ax.ravel(), files[:9]):
        img = S.load_upright(f)
        h, w = img.shape[:2]
        x0, y0, x1, y1 = S.ROIS[a.roi]
        crop = S.crop_rel(img, S.ROIS[a.roi])
        _, clean, _ = S.run_method(S.to_gray(crop), "adaptive", w / 3506.0)
        res = S.decide(S.extract_features(clean))
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = (0, 160, 0) if res.endswith("PRESENT") else (220, 0, 0)
        cv2.rectangle(vis, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), col, max(4, w // 250))
        axx.imshow(vis); axx.axis("off"); axx.set_title(os.path.basename(f)[:26] + "\n" + res, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.output, "figures", "overview_detection.png"), dpi=90)
    plt.close(fig)
    print("\nSelesai. Output di:", a.output)


if __name__ == "__main__":
    main()
