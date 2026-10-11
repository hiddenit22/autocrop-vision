"""
GUI Aplikasi Crop Otomatis Box pada Gambar
Antarmuka grafis menggunakan Tkinter
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from pathlib import Path

# Import modul utama
try:
    from crop_boxes import process_image, batch_process
except ImportError:
    messagebox.showerror("Error", "crop_boxes.py tidak ditemukan di direktori yang sama!")
    sys.exit(1)


class CropApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crop Otomatis Box Gambar")
        self.geometry("680x590")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")
        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        # Frame
        style.configure('Dark.TFrame', background='#1e1e2e')
        style.configure('Card.TFrame', background='#2a2a3e', relief='flat')
        # Label
        style.configure('Dark.TLabel', background='#1e1e2e', foreground='#cdd6f4',
                         font=('Segoe UI', 10))
        style.configure('Title.TLabel', background='#1e1e2e', foreground='#cba6f7',
                         font=('Segoe UI', 16, 'bold'))
        style.configure('Sub.TLabel', background='#1e1e2e', foreground='#6c7086',
                         font=('Segoe UI', 9))
        style.configure('Card.TLabel', background='#2a2a3e', foreground='#cdd6f4',
                         font=('Segoe UI', 10))
        # Entry
        style.configure('Dark.TEntry', fieldbackground='#313244', foreground='#cdd6f4',
                         insertcolor='#cdd6f4')
        # Combobox
        style.configure('Dark.TCombobox', fieldbackground='#313244', foreground='#cdd6f4',
                         background='#313244')
        # Button
        style.configure('Accent.TButton', background='#7c3aed', foreground='white',
                         font=('Segoe UI', 10, 'bold'), padding=(14, 8))
        style.map('Accent.TButton', background=[('active', '#6d28d9')])
        style.configure('Secondary.TButton', background='#313244', foreground='#cdd6f4',
                         font=('Segoe UI', 9), padding=(10, 6))
        style.map('Secondary.TButton', background=[('active', '#45475a')])
        # Progressbar
        style.configure('Purple.Horizontal.TProgressbar',
                         troughcolor='#313244', background='#7c3aed', thickness=8)
        # Checkbutton
        style.configure('Dark.TCheckbutton', background='#1e1e2e', foreground='#cdd6f4',
                         font=('Segoe UI', 10))

    def _build_ui(self):
        # ── Header ──────────────────────────────────────
        header = ttk.Frame(self, style='Dark.TFrame', padding=(24, 18, 24, 8))
        header.pack(fill='x')
        ttk.Label(header, text="🖼  Crop Otomatis Box Gambar", style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text="Deteksi & crop sel tabel/form secara otomatis menggunakan OpenCV",
                  style='Sub.TLabel').pack(anchor='w', pady=(2, 0))

        sep = tk.Frame(self, bg='#313244', height=1)
        sep.pack(fill='x', padx=24)

        # ── Body ────────────────────────────────────────
        body = ttk.Frame(self, style='Dark.TFrame', padding=(24, 14))
        body.pack(fill='both', expand=True)

        # Input Gambar
        self._row_field(body, "📁  Input Gambar / Folder:", 0)
        self.var_input = tk.StringVar()
        entry_in = ttk.Entry(body, textvariable=self.var_input, style='Dark.TEntry', width=46)
        entry_in.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        btn_file = ttk.Button(body, text="Pilih File", style='Secondary.TButton',
                              command=self._pick_file)
        btn_file.grid(row=1, column=2, padx=(6, 0), pady=(0, 4))
        btn_dir = ttk.Button(body, text="Pilih Folder", style='Secondary.TButton',
                             command=self._pick_folder)
        btn_dir.grid(row=1, column=3, padx=(4, 0), pady=(0, 4))

        # Output Folder
        self._row_field(body, "💾  Folder Output:", 2)
        self.var_output = tk.StringVar(value="output_crops")
        entry_out = ttk.Entry(body, textvariable=self.var_output, style='Dark.TEntry', width=46)
        entry_out.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 4))
        ttk.Button(body, text="Pilih Folder", style='Secondary.TButton',
                   command=self._pick_output).grid(row=3, column=2, padx=(6, 0), pady=(0, 4))

        body.columnconfigure(0, weight=1)
        ttk.Separator(body, orient='horizontal').grid(row=4, column=0, columnspan=4,
                                                       sticky='ew', pady=10)

        # ── Opsi ────────────────────────────────────────
        opt_frame = ttk.Frame(body, style='Card.TFrame', padding=12)
        opt_frame.grid(row=5, column=0, columnspan=4, sticky='ew', pady=(0, 10))

        # Metode
        ttk.Label(opt_frame, text="Metode Deteksi:", style='Card.TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 8))
        self.var_method = tk.StringVar(value='auto')
        combo = ttk.Combobox(opt_frame, textvariable=self.var_method,
                             values=['auto', 'grid', 'contour'],
                             state='readonly', width=12)
        combo.grid(row=0, column=1, sticky='w')

        # Padding
        ttk.Label(opt_frame, text="Padding (px):", style='Card.TLabel').grid(
            row=0, column=2, sticky='w', padx=(20, 8))
        self.var_padding = tk.IntVar(value=8)
        spin = tk.Spinbox(opt_frame, textvariable=self.var_padding,
                          from_=0, to=50, width=6,
                          bg='#313244', fg='#cdd6f4', insertbackground='#cdd6f4',
                          relief='flat', highlightthickness=1, highlightcolor='#45475a')
        spin.grid(row=0, column=3, sticky='w')

        # Min area
        ttk.Label(opt_frame, text="Min Area (px²):", style='Card.TLabel').grid(
            row=1, column=0, sticky='w', pady=(8, 0))
        self.var_min_area = tk.IntVar(value=5000)
        spin2 = tk.Spinbox(opt_frame, textvariable=self.var_min_area,
                           from_=100, to=100000, increment=500, width=8,
                           bg='#313244', fg='#cdd6f4', insertbackground='#cdd6f4',
                           relief='flat', highlightthickness=1, highlightcolor='#45475a')
        spin2.grid(row=1, column=1, sticky='w', pady=(8, 0))

        # Resize & size
        self.var_resize = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(opt_frame, text="Resize Output",
                               variable=self.var_resize, style='Dark.TCheckbutton')
        chk.grid(row=1, column=2, sticky='w', padx=(20, 0), pady=(8, 0))
        size_frame = ttk.Frame(opt_frame, style='Card.TFrame')
        size_frame.grid(row=1, column=3, sticky='w', pady=(8, 0))
        self.var_size_h = tk.IntVar(value=128)
        self.var_size_w = tk.IntVar(value=128)
        tk.Spinbox(size_frame, textvariable=self.var_size_h, from_=32, to=512, width=5,
                   bg='#313244', fg='#cdd6f4', insertbackground='#cdd6f4',
                   relief='flat').pack(side='left')
        tk.Label(size_frame, text="×", bg='#2a2a3e', fg='#cdd6f4').pack(side='left', padx=2)
        tk.Spinbox(size_frame, textvariable=self.var_size_w, from_=32, to=512, width=5,
                   bg='#313244', fg='#cdd6f4', insertbackground='#cdd6f4',
                   relief='flat').pack(side='left')

        # Debug & Batch
        self.var_debug = tk.BooleanVar(value=True)
        chk_debug = ttk.Checkbutton(opt_frame, text="Simpan Gambar Debug",
                                     variable=self.var_debug,
                                     style='Dark.TCheckbutton')
        chk_debug.grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.var_batch = tk.BooleanVar(value=False)
        chk_batch = ttk.Checkbutton(opt_frame, text="Mode Batch (folder)",
                                     variable=self.var_batch,
                                     style='Dark.TCheckbutton')
        chk_batch.grid(row=2, column=2, columnspan=2, sticky='w', pady=(8, 0))

        # Deskew & Crop to Content
        self.var_deskew = tk.BooleanVar(value=True)
        chk_deskew = ttk.Checkbutton(opt_frame, text="Luruskan Gambar (Deskew)",
                                      variable=self.var_deskew,
                                      style='Dark.TCheckbutton')
        chk_deskew.grid(row=3, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.var_crop_to_content = tk.BooleanVar(value=False)
        chk_crop_to_content = ttk.Checkbutton(opt_frame, text="Crop Fokus Karakter (Isi)",
                                               variable=self.var_crop_to_content,
                                               style='Dark.TCheckbutton')
        chk_crop_to_content.grid(row=3, column=2, columnspan=2, sticky='w', pady=(8, 0))

        # ── Progress ────────────────────────────────────
        self.progress = ttk.Progressbar(body, style='Purple.Horizontal.TProgressbar',
                                        mode='indeterminate', length=400)
        self.progress.grid(row=6, column=0, columnspan=4, sticky='ew', pady=(0, 6))

        # ── Log ─────────────────────────────────────────
        log_frame = tk.Frame(body, bg='#1e1e2e')
        log_frame.grid(row=7, column=0, columnspan=4, sticky='nsew')
        body.rowconfigure(7, weight=1)

        self.log_text = tk.Text(log_frame, bg='#11111b', fg='#a6e3a1',
                                 font=('Consolas', 9), relief='flat',
                                 height=6, wrap='word', state='disabled')
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # ── Footer Buttons ───────────────────────────────
        footer = ttk.Frame(self, style='Dark.TFrame', padding=(24, 10))
        footer.pack(fill='x')
        ttk.Button(footer, text="🚀  Mulai Crop", style='Accent.TButton',
                   command=self._start).pack(side='right')
        ttk.Button(footer, text="Bersihkan Log", style='Secondary.TButton',
                   command=self._clear_log).pack(side='right', padx=(0, 8))
        ttk.Button(footer, text="Buka Output", style='Secondary.TButton',
                   command=self._open_output).pack(side='left')

    def _row_field(self, parent, text, row):
        ttk.Label(parent, text=text, style='Dark.TLabel').grid(
            row=row, column=0, columnspan=4, sticky='w', pady=(6, 2))

    def _pick_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Semua", "*.*")])
        if path:
            self.var_input.set(path)

    def _pick_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.var_input.set(path)
            self.var_batch.set(True)

    def _pick_output(self):
        path = filedialog.askdirectory()
        if path:
            self.var_output.set(path)

    def _log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + "\n")
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    def _open_output(self):
        out = self.var_output.get()
        if os.path.isdir(out):
            os.startfile(out)
        else:
            messagebox.showinfo("Info", f"Folder tidak ditemukan: {out}")

    def _start(self):
        input_path = self.var_input.get().strip()
        output_path = self.var_output.get().strip()
        if not input_path:
            messagebox.showwarning("Peringatan", "Pilih gambar atau folder terlebih dahulu!")
            return
        if not output_path:
            messagebox.showwarning("Peringatan", "Tentukan folder output!")
            return

        # Redirect stdout ke log
        class LogRedirect:
            def __init__(self, callback):
                self.callback = callback
            def write(self, msg):
                if msg.strip():
                    self.callback(msg.strip())
            def flush(self):
                pass

        sys.stdout = LogRedirect(self._log)

        self.progress.start(10)
        thread = threading.Thread(target=self._run_crop, args=(input_path, output_path), daemon=True)
        thread.start()

    def _run_crop(self, input_path, output_path):
        try:
            kwargs = dict(
                method=self.var_method.get(),
                padding=self.var_padding.get(),
                resize_output=self.var_resize.get(),
                target_size=(self.var_size_h.get(), self.var_size_w.get()),
                min_area=self.var_min_area.get(),
                save_debug=self.var_debug.get(),
                deskew=self.var_deskew.get(),
                crop_to_content=self.var_crop_to_content.get(),
            )
            if self.var_batch.get():
                batch_process(input_path, output_path, **kwargs)
            else:
                process_image(input_path, output_path, **kwargs)

            self.after(0, lambda: messagebox.showinfo("Selesai", "Crop selesai! Cek folder output."))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            sys.stdout = sys.__stdout__
            self.after(0, self.progress.stop)


if __name__ == '__main__':
    app = CropApp()
    app.mainloop()
