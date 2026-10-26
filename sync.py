"""
Sync dist/WhisperXTranscriber → dist/release/WhisperXTranscriber
Preserves runtime/ and Models/ in the destination (installed ML packages).
Run after every build: python sync.py
"""
import shutil
import pathlib

src = pathlib.Path("dist/WhisperXTranscriber")
dst = pathlib.Path("dist/release/WhisperXTranscriber")
SKIP = {"runtime", "Models"}

dst.mkdir(parents=True, exist_ok=True)
updated = []

for item in src.iterdir():
    if item.name in SKIP:
        continue
    t = dst / item.name
    if item.is_dir():
        if t.exists():
            shutil.rmtree(t)
        shutil.copytree(item, t)
    else:
        shutil.copy2(item, t)
    updated.append(item.name)

print("Synced:", ", ".join(sorted(updated)))
