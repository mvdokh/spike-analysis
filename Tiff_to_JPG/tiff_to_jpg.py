"""
Convert TIFF images to uncompressed (max quality) JPG copies.
JPGs are written alongside the originals with the same basename.
"""

from pathlib import Path

from PIL import Image

# Keyence mosaics are huge (~0.5–1B pixels); disable Pillow's DOS safeguard.
Image.MAX_IMAGE_PIXELS = None

SOURCE_DIRS = [
    Path(r"D:\Keyence\axonal_tracing_1"),
    Path(r"D:\Keyence\RG1"),
    Path(r"D:\Keyence\RG2"),
]

TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}


def to_jpg_compatible(img: Image.Image) -> Image.Image:
    """Convert modes that JPEG cannot store (e.g. 16-bit, RGBA, palette)."""
    if img.mode in ("RGB", "L"):
        return img
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode == "P":
        return img.convert("RGBA").convert("RGB") if "transparency" in img.info else img.convert("RGB")
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        # Scale 16-bit to 8-bit for JPEG
        arr = img.point(lambda x: x * (255.0 / 65535.0)).convert("L")
        return arr
    return img.convert("RGB")


def convert_tiff(tiff_path: Path) -> Path:
    jpg_path = tiff_path.with_suffix(".jpg")
    with Image.open(tiff_path) as img:
        # Multi-page TIFFs: export first page (typical for Keyence stills)
        img.seek(0)
        compatible = to_jpg_compatible(img)
        # quality=100, subsampling=0, optimize=False ≈ no intentional compression
        compatible.save(
            jpg_path,
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
        )
    return jpg_path


def main() -> None:
    converted = 0
    skipped = 0
    errors = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"SKIP (missing): {source_dir}")
            continue

        tiffs = sorted(
            p for p in source_dir.rglob("*")
            if p.is_file() and p.suffix in TIFF_EXTENSIONS
        )
        print(f"\n{source_dir} — {len(tiffs)} TIFF(s)")

        for tiff_path in tiffs:
            jpg_path = tiff_path.with_suffix(".jpg")
            if jpg_path.exists():
                print(f"  exists, skip: {jpg_path.name}")
                skipped += 1
                continue
            try:
                out = convert_tiff(tiff_path)
                print(f"  OK: {tiff_path.name} -> {out.name}")
                converted += 1
            except Exception as exc:
                print(f"  FAIL: {tiff_path} ({exc})")
                errors += 1

    print(f"\nDone. converted={converted}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
