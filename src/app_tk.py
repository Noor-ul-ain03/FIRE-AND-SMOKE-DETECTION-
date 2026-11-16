"""Module 9b: desktop interface (no extra install: tkinter comes with Python).
Usage: python src/app_tk.py
Click 'Choose a photo', then read the probabilities and the decision.
"""
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageOps, ImageTk

from predict import predict

DISCLAIMER = ("Decision-support only. This model never saw photos without fire or smoke, "
              "so it can raise false alarms. It must not replace certified fire alarms or a person.")
COLOURS = {"FIRE + SMOKE": "#b00020", "FIRE": "#b00020", "SMOKE": "#b26a00", "NONE DETECTED": "#2e7d32"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fire and Smoke Detector")
        self.geometry("760x780")
        self.photo = None
        tk.Button(self, text="Choose a photo...", font=("Segoe UI", 12),
                  command=self.choose).pack(pady=10)
        self.preview = tk.Label(self)
        self.preview.pack()
        self.decision = tk.Label(self, text="", font=("Segoe UI", 20, "bold"))
        self.decision.pack(pady=(12, 2))
        self.details = tk.Label(self, text="", font=("Segoe UI", 12), justify="left")
        self.details.pack()
        tk.Label(self, text=DISCLAIMER, fg="gray", wraplength=700,
                 font=("Segoe UI", 9)).pack(side="bottom", pady=10)

    def choose(self):
        path = filedialog.askopenfilename(
            title="Choose a photo", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path)
            shown = ImageOps.exif_transpose(img).convert("RGB")
            shown.thumbnail((560, 400))
            self.photo = ImageTk.PhotoImage(shown)
            self.preview.config(image=self.photo)
            self.decision.config(text="Working...", fg="gray")
            self.details.config(text="")
            self.update_idletasks()
            r = predict(img)
        except Exception as error:
            messagebox.showerror("Error", str(error))
            self.decision.config(text="")
            return
        self.decision.config(text=r["decision"], fg=COLOURS.get(r["decision"], "black"))
        lines = [f"Fire probability:   {r['p_fire']:.2f}   (threshold {r['thr_fire']:.2f})",
                 f"Smoke probability: {r['p_smoke']:.2f}   (threshold {r['thr_smoke']:.2f})"]
        if r["needs_human_check"]:
            lines.append("NEEDS HUMAN CHECK: a probability is close to its threshold.")
        self.details.config(text="\n".join(lines))


if __name__ == "__main__":
    App().mainloop()
