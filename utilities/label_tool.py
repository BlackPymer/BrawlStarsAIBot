import csv
import re
import sys
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

IMAGES_DIR = Path("health_dataset/images")
LABELS_CSV = Path("health_dataset/labels.csv")
PREDICTIONS_CSV = Path("health_dataset/labels_with_predictions.csv")


def parse_frame(fname: str) -> int:
    m = re.search(r'frame_?(\d+)', fname)
    return int(m.group(1)) if m else 0


def build_prediction_map():
    pred = {}
    with open(PREDICTIONS_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            hp = r["hp"].strip()
            if hp:
                pred[r["filename"]] = hp
    return pred


class LabelTool:
    def __init__(self, start_idx):
        self.predictions = build_prediction_map()

        with open(LABELS_CSV) as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        self.idx = start_idx
        self.total = len(self.rows)
        self.labeled = sum(1 for r in self.rows if r["hp"].strip())

        self.root = tk.Tk()
        self.root.title("Label Tool")
        self.root.geometry("500x400")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=5)

        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=2)

        self.info_var = tk.StringVar()
        tk.Label(info_frame, textvariable=self.info_var, font=("Arial", 9)).pack()

        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="HP:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        self.hp_var = tk.StringVar()
        self.hp_entry = tk.Entry(input_frame, textvariable=self.hp_var, font=("Arial", 14), width=12)
        self.hp_entry.pack(side=tk.LEFT, padx=5)
        self.hp_entry.bind("<Return>", self.on_enter)
        self.hp_entry.bind("<Escape>", lambda e: self.root.quit())
        self.hp_entry.bind("<BackSpace>", self.on_backspace)
        self.hp_entry.focus_set()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Сохранить →", command=self.on_enter, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="← Назад", command=self.go_back, width=10).pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.progress_var, font=("Arial", 8)).pack()

        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 8), fg="gray").pack()

        self.show_current()

    def show_current(self):
        if self.idx >= self.total:
            self.status_var.set("Готово! Все размечено.")
            return

        row = self.rows[self.idx]
        fname = row["filename"]
        gt = row["hp"].strip()
        pred = self.predictions.get(fname, "")

        img_path = IMAGES_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            self.status_var.set(f"Файл не найден: {fname}")
            self.idx += 1
            self.root.after(100, self.show_current)
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = min(300 / h, 400 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.image_label.config(image=self.tk_img)

        short = fname.replace("Запись экрана 2026-07-07 121042_", "")
        self.info_var.set(f"[{self.idx + 1}/{self.total}]  {short}")
        self.status_var.set(f"GT: {gt or '?'}  |  Pred: {pred or '?'}")

        self.hp_var.set(pred if pred else gt)
        self.hp_entry.selection_range(0, tk.END)
        self.hp_entry.focus_set()

        labeled_now = sum(1 for r in self.rows if r["hp"].strip())
        self.progress_var.set(f"Размечено: {labeled_now - self.labeled} за сессию  |  Всего: {labeled_now}/{self.total}")

    def save(self):
        with open(LABELS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "hp"])
            writer.writeheader()
            writer.writerows(self.rows)

    def on_enter(self, event=None):
        val = self.hp_var.get().strip()
        if val:
            self.rows[self.idx]["hp"] = val
        self.save()
        self.idx += 1
        self.show_current()

    def go_back(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_current()

    def on_backspace(self, event):
        if self.hp_var.get() == "":
            self.go_back()
        return None

    def run(self):
        self.root.mainloop()


def main():
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    start_arg = sys.argv[1] if len(sys.argv) > 1 else None
    start_idx = 0

    if start_arg:
        target_frame = int(start_arg)
        for i, r in enumerate(rows):
            if parse_frame(r["filename"]) >= target_frame:
                start_idx = i
                break
    else:
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["hp"].strip():
                start_idx = i + 1
                break

    if start_idx >= len(rows):
        print("Всё уже размечено!")
        return

    labeled = sum(1 for r in rows if r["hp"].strip())
    total = len(rows)
    print(f"Размечено: {labeled}/{total}  Старт со строки {start_idx + 1}")

    app = LabelTool(start_idx)
    app.run()


if __name__ == "__main__":
    main()
