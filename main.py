from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytesseract


# =========================================================
# 1. KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Folder input sesuai struktur proyek
NOMOR_DIR = BASE_DIR / "Nomor_Ijazah"
TANDA_TANGAN_DIR = BASE_DIR / "Tanda_tangan"

# Folder output
OUTPUT_DIR = BASE_DIR / "hasil"
KOSONG_DIR = OUTPUT_DIR / "Tanda_Tangan_Kosong"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sesuaikan dengan nomor ijazah yang sebenarnya.
# Jika setiap gambar memiliki nomor berbeda, gunakan ground truth
# masing-masing gambar agar perhitungan CER akurat.
GROUND_TRUTH_NUMBER = "571012022000056"

# Lokasi program Tesseract OCR di Windows
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# =========================================================
# 2. UTILITAS BACA CITRA DAN URUTAN FILE
# =========================================================
def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_images(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {folder}"
        )

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ],
        key=natural_key
    )


def read_gray(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Citra gagal dibaca: {path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return image, gray


# =========================================================
# 3. METODE IMAGE ENHANCEMENT
# =========================================================
def brightness_adjustment(gray, beta=25):
    """
    Meningkatkan kecerahan citra.
    """
    return cv2.convertScaleAbs(
        gray,
        alpha=1.0,
        beta=beta
    )


def contrast_stretching(gray):
    """
    Meningkatkan kontras berdasarkan persentil intensitas.
    """
    low, high = np.percentile(
        gray,
        (2, 98)
    )

    if high <= low:
        return gray.copy()

    result = (
        gray.astype(np.float32) - low
    ) * (255.0 / (high - low))

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def histogram_equalization(gray):
    """
    Memperbaiki distribusi intensitas citra.
    """
    return cv2.equalizeHist(gray)


# =========================================================
# 4. OCR DAN CHARACTER ERROR RATE
# =========================================================
def perform_ocr(gray):
    """
    Membaca nomor ijazah menggunakan Tesseract OCR.
    """
    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=0123456789"
    )

    text = pytesseract.image_to_string(
        gray,
        config=config
    )

    # Mengambil karakter angka saja
    return re.sub(r"\D", "", text)


def edit_distance(reference, prediction):
    """
    Menghitung Levenshtein Distance.
    """
    previous = list(
        range(len(prediction) + 1)
    )

    for i, ref_char in enumerate(
        reference,
        start=1
    ):
        current = [i]

        for j, pred_char in enumerate(
            prediction,
            start=1
        ):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            substitution = (
                previous[j - 1]
                + (ref_char != pred_char)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


def character_error_rate(reference, prediction):
    """
    CER = jumlah kesalahan karakter / jumlah karakter referensi.
    """
    if not reference:
        raise ValueError(
            "Ground truth tidak boleh kosong."
        )

    return (
        edit_distance(reference, prediction)
        / len(reference)
    )


# =========================================================
# 5. DETEKSI TANDA TANGAN
# =========================================================
def detect_signature(
    gray,
    min_ink_pixels=80,
    min_ink_ratio=0.001
):
    """
    Mendeteksi keberadaan tanda tangan berdasarkan
    piksel tinta setelah thresholding dan morfologi.
    """

    # Mengurangi noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Thresholding Otsu
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    # Menghilangkan noise kecil
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    opened = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Menghubungkan goresan yang terputus
    cleaned = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Analisis komponen yang saling terhubung
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )

    filtered = np.zeros_like(cleaned)

    for component_id in range(1, count):
        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        if area >= 8:
            filtered[
                labels == component_id
            ] = 255

    # Menghitung piksel tinta
    ink_pixels = int(
        cv2.countNonZero(filtered)
    )

    total_pixels = (
        filtered.shape[0]
        * filtered.shape[1]
    )

    ink_ratio = (
        ink_pixels / max(total_pixels, 1)
    )

    present = (
        ink_pixels >= min_ink_pixels
        and ink_ratio >= min_ink_ratio
    )

    status = (
        "SIGNATURE PRESENT"
        if present
        else "SIGNATURE ABSENT"
    )

    return (
        status,
        binary,
        opened,
        filtered,
        ink_pixels,
        ink_ratio
    )


# =========================================================
# 6. MEMBUAT CONTOH CITRA TANDA TANGAN KOSONG
# =========================================================
def create_blank_examples(reference_images):
    """
    Membuat sembilan citra kosong sintetis berwarna putih.
    Citra ini digunakan untuk menguji deteksi tanda tangan,
    bukan sebagai pengganti foto ijazah kosong yang asli.
    """

    KOSONG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    blank_paths = []

    if not reference_images:
        print(
            "Tidak ada citra tanda tangan "
            "untuk menentukan ukuran citra kosong."
        )

        return blank_paths

    for index, path in enumerate(
        reference_images[:9],
        start=1
    ):
        _, gray = read_gray(path)

        blank = np.full(
            gray.shape,
            255,
            dtype=np.uint8
        )

        blank_path = (
            KOSONG_DIR
            / f"kosong_{index:02d}.png"
        )

        if not cv2.imwrite(
            str(blank_path),
            blank
        ):
            raise OSError(
                f"Gagal menyimpan citra kosong: {blank_path}"
            )

        blank_paths.append(blank_path)

    return blank_paths


# =========================================================
# 7. VISUALISASI DAN ANALISIS
# =========================================================
def visualize_one(
    index,
    number_path,
    signature_path
):
    original, gray = read_gray(
        number_path
    )

    # Enhancement nomor ijazah
    methods = {
        "Brightness Adjustment":
            brightness_adjustment(gray),

        "Contrast Stretching":
            contrast_stretching(gray),

        "Histogram Equalization":
            histogram_equalization(gray),
    }

    # OCR setiap metode
    ocr_results = {
        name: perform_ocr(img)
        for name, img in methods.items()
    }

    # Perhitungan CER
    cer_results = {
        name: character_error_rate(
            GROUND_TRUTH_NUMBER,
            text
        )
        for name, text in ocr_results.items()
    }

    # Menentukan metode dengan CER terendah
    chosen_method = min(
        cer_results,
        key=cer_results.get
    )

    chosen_text = ocr_results[
        chosen_method
    ]

    chosen_cer = cer_results[
        chosen_method
    ]

    # Deteksi tanda tangan
    _, signature_gray = read_gray(
        signature_path
    )

    (
        sig_status,
        sig_binary,
        sig_opened,
        sig_cleaned,
        ink_pixels,
        ink_ratio
    ) = detect_signature(
        signature_gray
    )

    # Menyimpan hasil enhancement
    for name, enhanced in methods.items():
        safe_name = (
            name.lower().replace(" ", "_")
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"{number_path.stem}_{safe_name}.png"
            ),
            enhanced
        )

    # Menyimpan hasil pemrosesan tanda tangan
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{signature_path.stem}_threshold.png"
        ),
        sig_binary
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{signature_path.stem}_opening.png"
        ),
        sig_opened
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{signature_path.stem}_morphology.png"
        ),
        sig_cleaned
    )

    # Visualisasi hasil
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8)
    )

    fig.suptitle(
        f"Verifikasi Ijazah {index} - "
        f"{number_path.name}",
        fontsize=14,
        fontweight="bold"
    )

    panels = [
        (
            "Citra Asli",
            cv2.cvtColor(
                original,
                cv2.COLOR_BGR2RGB
            )
        ),
        (
            "Grayscale",
            gray
        ),
        (
            "Brightness Adjustment",
            methods["Brightness Adjustment"]
        ),
        (
            "Contrast Stretching",
            methods["Contrast Stretching"]
        ),
        (
            "Histogram Equalization",
            methods["Histogram Equalization"]
        ),
        (
            "Deteksi Tanda Tangan",
            sig_cleaned
        ),
    ]

    for ax, (title, img) in zip(
        axes.flat,
        panels
    ):
        ax.imshow(
            img,
            cmap="gray" if img.ndim == 2 else None
        )

        ax.set_title(title)
        ax.axis("off")

    fig.text(
        0.03,
        0.015,
        f"OCR: {chosen_text or 'TIDAK TERBACA'} | "
        f"Metode CER terendah: {chosen_method} | "
        f"CER: {chosen_cer:.2%}\n"
        f"Tanda tangan: {sig_status} | "
        f"Piksel tinta: {ink_pixels} | "
        f"Rasio tinta: {ink_ratio:.5f}",
        fontsize=10,
        va="bottom"
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 0.93]
    )

    fig.savefig(
        OUTPUT_DIR
        / f"visualisasi_verifikasi_{index:02d}.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Menyusun data untuk CSV
    rows = []

    for name, text in ocr_results.items():
        rows.append({
            "No": index,
            "Citra Nomor": number_path.name,
            "Citra Tanda Tangan": signature_path.name,
            "Metode Enhancement": name,
            "Hasil OCR": text,
            "Ground Truth": GROUND_TRUTH_NUMBER,
            "CER": cer_results[name],
            "Tanda Tangan": sig_status,
            "Piksel Tinta": ink_pixels,
            "Rasio Tinta": ink_ratio,
        })

    # Menampilkan hasil pada terminal
    print(
        f"\n===== VERIFIKASI {index} ====="
    )

    print(
        f"File nomor: {number_path.name}"
    )

    print(
        f"File tanda tangan: {signature_path.name}"
    )

    print(
        f"Nomor Ijazah OCR: "
        f"{chosen_text or 'TIDAK TERBACA'}"
    )

    print(
        f"Metode terbaik berdasarkan CER: "
        f"{chosen_method}"
    )

    print(
        f"CER metode terpilih: {chosen_cer:.2%}"
    )

    for name, score in cer_results.items():
        print(
            f"CER {name}: {score:.2%}"
        )

    print(
        f"Tanda tangan: {sig_status}"
    )

    return rows


# =========================================================
# 8. UJI CITRA TANDA TANGAN KOSONG
# =========================================================
def test_blank_signatures(blank_paths):
    results = []

    for path in blank_paths:
        _, gray = read_gray(path)

        (
            status,
            _,
            _,
            _,
            ink_pixels,
            ink_ratio
        ) = detect_signature(gray)

        results.append({
            "Citra Kosong": path.name,
            "Hasil Deteksi": status,
            "Piksel Tinta": ink_pixels,
            "Rasio Tinta": ink_ratio,
            "Sesuai Harapan":
                status == "SIGNATURE ABSENT",
        })

    if results:
        output_csv = (
            OUTPUT_DIR
            / "pengujian_9_citra_kosong.csv"
        )

        pd.DataFrame(
            results
        ).to_csv(
            output_csv,
            index=False,
            encoding="utf-8-sig"
        )

        correct = sum(
            row["Sesuai Harapan"]
            for row in results
        )

        print(
            "\n===== UJI CITRA KOSONG SINTETIS ====="
        )

        for row in results:
            print(
                f"{row['Citra Kosong']}: "
                f"{row['Hasil Deteksi']}"
            )

        print(
            f"Deteksi ABSENT sesuai harapan: "
            f"{correct}/{len(results)}"
        )

        print(
            f"Rekap: {output_csv}"
        )

    return results


# =========================================================
# 9. PROGRAM UTAMA
# =========================================================
def main():
    # Memastikan Tesseract dapat digunakan
    try:
        print(
            "Lokasi Tesseract:",
            pytesseract.pytesseract.tesseract_cmd
        )

        print(
            "Versi Tesseract:",
            pytesseract.get_tesseract_version()
        )

    except Exception as error:
        raise RuntimeError(
            "Tesseract tidak dapat dijalankan. "
            "Install Tesseract OCR dan sesuaikan "
            "TESSERACT_EXE pada main.py."
        ) from error

    # Membaca gambar dari folder input
    number_images = get_images(
        NOMOR_DIR
    )

    signature_images = get_images(
        TANDA_TANGAN_DIR
    )

    print("\n===== DATA INPUT =====")

    print(
        "Folder nomor:",
        NOMOR_DIR
    )

    print(
        "Jumlah citra nomor:",
        len(number_images)
    )

    print(
        "Folder tanda tangan:",
        TANDA_TANGAN_DIR
    )

    print(
        "Jumlah citra tanda tangan:",
        len(signature_images)
    )

    if not number_images:
        raise ValueError(
            f"Folder {NOMOR_DIR} tidak berisi gambar."
        )

    if not signature_images:
        raise ValueError(
            f"Folder {TANDA_TANGAN_DIR} tidak berisi gambar."
        )

    if len(number_images) != len(signature_images):
        raise ValueError(
            "Jumlah gambar pada kedua folder berbeda. "
            "Pastikan pasangan file sesuai sebelum "
            "menjalankan analisis."
        )

    # Memproses pasangan gambar berdasarkan urutan nama file.
    all_rows = []

    for index, (
        number_path,
        signature_path
    ) in enumerate(
        zip(
            number_images,
            signature_images
        ),
        start=1
    ):
        all_rows.extend(
            visualize_one(
                index,
                number_path,
                signature_path
            )
        )

    # Menyimpan tabel perbandingan OCR
    comparison_path = (
        OUTPUT_DIR
        / "perbandingan_enhancement_ocr.csv"
    )

    pd.DataFrame(
        all_rows
    ).to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Menghitung CER agregat setiap metode
    print(
        "\n===== CER AGREGAT PER METODE ====="
    )

    total_reference_characters = (
        len(GROUND_TRUTH_NUMBER)
        * len(number_images)
    )

    for method in (
        "Brightness Adjustment",
        "Contrast Stretching",
        "Histogram Equalization",
    ):
        method_rows = [
            row for row in all_rows
            if row["Metode Enhancement"] == method
        ]

        total_errors = sum(
            edit_distance(
                row["Ground Truth"],
                row["Hasil OCR"]
            )
            for row in method_rows
        )

        aggregate_cer = (
            total_errors
            / max(total_reference_characters, 1)
        )

        print(
            f"{method}: {aggregate_cer:.2%}"
        )

    # Membuat dan menguji citra tanda tangan kosong
    blank_paths = create_blank_examples(
        signature_images
    )

    test_blank_signatures(
        blank_paths
    )

    print("\n===== SELESAI =====")

    print(
        "Semua hasil disimpan di:",
        OUTPUT_DIR
    )

    print(
        "Tabel OCR:",
        comparison_path
    )


if __name__ == "__main__":
    main()