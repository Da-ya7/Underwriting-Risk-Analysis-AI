"""
OCR preprocessing: deskew, multi-variant thresholding, per-word confidence
gating, region re-OCR for low-confidence tokens.

Why: single global-Otsu threshold on full card = one bad local region
(uneven lighting, embossed font, watermark) corrupts unrelated text
elsewhere. Fix = try several preprocess variants, keep per-word confidence,
and re-OCR (crop+upscale) any word tesseract wasn't sure about.
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

LOW_CONF_THRESHOLD = 55  # tesseract conf 0-100; below this -> re-OCR that word
MIN_UPSCALE_TARGET = 1500


def _resize_min(gray, target=MIN_UPSCALE_TARGET):
    h, w = gray.shape
    if max(h, w) < target:
        scale = target / max(h, w)
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def deskew(gray):
    """Rotate so text baseline is horizontal. Uses the bounding box of dark
    pixels (text/lines) rather than the whole card edge, since card borders
    are often not the same angle as the printed text block."""
    inv = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if coords.shape[0] < 50:
        return gray  # not enough signal, skip
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray  # ignore noise angles / bad detections
    (h, w) = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _variant_otsu(gray):
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _variant_adaptive(gray):
    denoised = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def _variant_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _variant_illum_norm(gray):
    """Illumination normalization: divide image by its own heavily-blurred
    version. Removes glare streaks / uneven lighting / watermark texture in
    one shot — none of that survives dividing it out, since it's low-freq
    by nature, while ink strokes are high-freq and stay. Fixes the case
    other 3 variants can't: physical glare/scratch sitting ON TOP of a text
    line, which bilateral/CLAHE/adaptive all treat as a contrast problem
    (they aren't) instead of an illumination problem (it is)."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    norm = cv2.divide(gray, bg, scale=255)
    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def build_variants(img):
    """img: BGR. Returns list of (name, processed_gray) preprocess variants."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _resize_min(gray)
    gray = deskew(gray)
    return [
        ("otsu", _variant_otsu(gray)),
        ("adaptive", _variant_adaptive(gray)),
        ("clahe", _variant_clahe(gray)),
        ("illum_norm", _variant_illum_norm(gray)),
    ]


def ocr_with_confidence(processed, lang="eng", psm=6):
    """Runs tesseract, returns (text, mean_confidence, low_conf_words)
    where low_conf_words = [(text, conf, (x,y,w,h)), ...]"""
    data = pytesseract.image_to_data(
        processed, lang=lang, config=f"--psm {psm}", output_type=Output.DICT
    )
    words, confs, low_conf = [], [], []
    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if not word:
            continue
        words.append(word)
        if conf >= 0:
            confs.append(conf)
            if conf < LOW_CONF_THRESHOLD:
                box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                low_conf.append((word, conf, box, i))
    text = pytesseract.image_to_string(processed, lang=lang, config=f"--psm {psm}")
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf, low_conf, data


def reocr_low_confidence_words(processed, low_conf_words, lang="eng", pad=8):
    """For each low-confidence word, crop a padded box around it, upscale
    3x, and re-OCR in isolation (psm 8 = single word). Isolation removes
    interference from neighbouring watermark texture / adjacent glyphs
    that dragged the whole-line confidence down. Returns {index: better_word}
    for words where the re-OCR attempt came back non-empty and alphanumeric."""
    h_img, w_img = processed.shape[:2]
    corrections = {}
    for word, conf, (x, y, w, h), idx in low_conf_words:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = processed[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        retry_text = pytesseract.image_to_string(
            crop, lang=lang, config="--psm 8"
        ).strip()
        if retry_text and retry_text != word:
            corrections[idx] = retry_text
    return corrections


def run_ocr_pipeline(img, lang="eng", psm=6):
    """Full pipeline: multi-variant OCR + confidence-gated region re-OCR.
    Returns dict: {variant_name: {"text", "mean_conf", "corrected_text"}}
    plus "best_variant" name (highest mean_conf)."""
    results = {}
    for name, processed in build_variants(img):
        text, mean_conf, low_conf, data = ocr_with_confidence(processed, lang=lang, psm=psm)
        corrected_text = text
        if low_conf:
            corrections = reocr_low_confidence_words(processed, low_conf, lang=lang)
            if corrections:
                tokens = [data["text"][i] for i in range(len(data["text"]))]
                for idx, better in corrections.items():
                    tokens[idx] = better
                # rebuild line-preserving text: group by (block,par,line)
                corrected_text = _rebuild_text(data, tokens)
        results[name] = {"text": text, "mean_conf": mean_conf, "corrected_text": corrected_text}

    best_variant = max(results, key=lambda k: results[k]["mean_conf"])
    return results, best_variant


def _rebuild_text(data, tokens):
    lines = {}
    n = len(tokens)
    for i in range(n):
        if not tokens[i].strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(tokens[i])
    ordered_keys = sorted(lines.keys())
    return "\n".join(" ".join(lines[k]) for k in ordered_keys)