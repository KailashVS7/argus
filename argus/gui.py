from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from .inference import run_inference
from .modeling import build_model


class ArgusGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Argus Proto 1")
        self.geometry("600x180")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "outputs"))

        tk.Label(self, text="Input image (TIFF/GeoTIFF or RGB demo image):").pack(anchor="w", padx=10, pady=(10, 2))
        row = tk.Frame(self)
        row.pack(fill="x", padx=10)
        tk.Entry(row, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse", command=self._browse_input).pack(side="left", padx=4)

        tk.Label(self, text="Output directory:").pack(anchor="w", padx=10, pady=(8, 2))
        row2 = tk.Frame(self)
        row2.pack(fill="x", padx=10)
        tk.Entry(row2, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        tk.Button(row2, text="Browse", command=self._browse_output).pack(side="left", padx=4)

        tk.Button(self, text="Run CPU Inference", command=self._run).pack(pady=12)

    def _browse_input(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg")])
        if file_path:
            self.input_path.set(file_path)

    def _browse_output(self) -> None:
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)

    def _run(self) -> None:
        path = self.input_path.get().strip()
        if not path:
            messagebox.showerror("Argus", "Please choose an input file")
            return
        model, _ = build_model(weights="none", no_download=True)
        payload = run_inference(
            model=model,
            vv_path=None,
            vh_path=None,
            input_path=path,
            output_dir=self.output_dir.get(),
        )
        messagebox.showinfo("Argus", f"Saved {len(payload['detections'])} detections to {self.output_dir.get()}")


def launch_gui() -> None:
    app = ArgusGui()
    app.mainloop()
