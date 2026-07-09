import cv2
from pathlib import Path

VIDEO = Path("videos/Запись экрана 2026-07-07 121042.mp4")
LABEL_DIR = Path("datasets/datasetV2/obj_Train_data")
IMG_DIR = Path("datasets/datasetV2/images/train")
LBL_DIR = Path("datasets/datasetV2/labels/train")

IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

frame_nums = sorted({
    int(f.stem.split("_")[1]) for f in LABEL_DIR.glob("*.txt")
})
print(f"Found {len(frame_nums)} frames to extract")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    print(f"Error: cannot open {VIDEO}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video has {total_frames} frames")

extracted = 0
for frame_idx in frame_nums:
    if frame_idx >= total_frames:
        print(f"Frame {frame_idx} exceeds video length ({total_frames}), skipping")
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        continue
    img_path = IMG_DIR / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(img_path), img)
    lbl_src = LABEL_DIR / f"frame_{frame_idx:06d}.txt"
    lbl_dst = LBL_DIR / f"frame_{frame_idx:06d}.txt"
    if lbl_src.exists():
        lbl_dst.write_text(lbl_src.read_text())
    extracted += 1
    if extracted % 200 == 0:
        print(f"Extracted {extracted}/{len(frame_nums)}")

cap.release()
print(f"Done – extracted {extracted} images to {IMG_DIR}")
