from pathlib import Path
from PIL import Image

src = Path("raw")
dst = Path("tags")
dst.mkdir(exist_ok=True)

for path in src.glob("*.png"):
    img = Image.open(path).convert("L")
    img = img.resize((1024, 1024), resample=Image.NEAREST)
    img = img.point(lambda p: 255 if p > 128 else 0)
    img.save(dst / path.name)