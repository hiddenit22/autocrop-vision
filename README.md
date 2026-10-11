# 🖼️ AutoCrop Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://palletsprojects.com/p/flask/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**AutoCrop Vision** is an automated computer vision application designed to detect, crop, and organize individual cells and boxes from scanned forms, handwriting tables, survey sheets, and grid-based documents with pixel-level precision.

It offers a modern **Web App** (Flask REST API + interactive canvas), a lightweight **Desktop GUI** (Tkinter), and a high-performance **CLI** for batch processing.

---

## ✨ Features

- **🌐 Modern Web Interface:**
  - Interactive canvas preview with zoom levels (850px, 1100px, 1450px, 1800px, 1:1).
  - Pre-crop confirmation: visual overlay of detected/template boxes before cutting.
  - Per-image box adjustments (`exact` mode) and one-click recropping with automatic replacement.
  - Client-side folder batch processing (process hundreds or thousands of files in chunks without overwhelming browser memory).
  - Reusable template presets (save/load custom box layouts).
- **🖥️ Desktop GUI (Tkinter):**
  - Standalone desktop application for quick offline cropping and previewing.
- **⚡ High-Performance CLI:**
  - Command-line tools for automated server tasks and headless batch jobs.
- **🧠 Advanced Detection Algorithms:**
  - **`auto`**: Automatic morphological horizontal & vertical line detection with fallback to contour analysis.
  - **`manual`**: ROI-guided detection with intelligent snap-to-line alignment.
  - **`exact`**: Strict pixel-coordinate extraction without distortion or snapping.
- **🔄 Auto-Deskew & Content Snapping:**
  - Automatic paper rotation correction (deskew).
  - Optional tight cropping around handwritten characters inside each box (`--crop-to-content`).
- **📁 Organized Output Management:**
  - Organizes output into categorized folders (`Box 1/`, `Box 2/`, etc.) with systematic file suffixes (`_a`, `_b`, `_c`...) and debug timestamps.

---

## 📁 Project Structure

```text
autocrop-vision/
├── app.py                      # Flask REST API backend server & Web UI routes
├── crop_boxes.py               # Core computer vision engine (OpenCV processing & CLI)
├── gui_crop.py                 # Tkinter desktop graphical interface
├── templates/
│   └── index.html              # Responsive Web UI (Interactive canvas & Dark mode)
├── templates.json              # Saved box templates configuration
├── requirements.txt            # Python dependencies
├── MENU_APLIKASI.bat           # Interactive Windows console menu launcher
├── JALANKAN_WEB_APP.bat        # 1-click launcher for Web Application
├── JALANKAN_GUI_DESKTOP.bat    # 1-click launcher for Tkinter GUI
└── README.md                   # Project documentation
```

---

## ⚙️ Installation Guide

### 1. Prerequisites
Ensure you have **Python 3.8** or newer installed on your machine.
Verify your Python installation:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone git@github.com:hiddenit22/autocrop-vision.git
cd autocrop-vision
```
*(Or use HTTPS: `git clone https://github.com/hiddenit22/autocrop-vision.git`)*

### 3. Create a Virtual Environment (Recommended)

- **On Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
Install all required Python packages using pip:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Web Application (Recommended)

1. Start the Flask local server:
   ```bash
   python app.py
   ```
   *(Or double-click `JALANKAN_WEB_APP.bat` on Windows)*

2. Open your web browser and navigate to:
   ```text
   http://localhost:5050
   ```

3. **Workflow in Web App:**
   - Choose your source image(s) or select a local folder.
   - Adjust target dimensions (e.g., `128x128`), inner margins, and resolution.
   - Click **Preview Kotak** to visually inspect boxes on top of the document.
   - Click **Jalankan Pemotongan (Crop)** to process.
   - Review results and use **Sesuaikan Kotak & Crop Ulang** if any individual cell needs fine-tuning.

---

### Option 2: Desktop GUI (Tkinter)

For quick, offline desktop operation:
```bash
python gui_crop.py
```
*(Or double-click `JALANKAN_GUI_DESKTOP.bat` on Windows)*

---

### Option 3: Command Line Interface (CLI)

Run `crop_boxes.py` directly for script automation or batch processing:

- **Process a single document:**
  ```bash
  python crop_boxes.py -i input_form.jpg -o output_directory/
  ```

- **Batch process an entire directory:**
  ```bash
  python crop_boxes.py -i path/to/images_folder/ -o output_directory/ --batch
  ```

- **Custom output resolution (e.g., 128x128 pixels):**
  ```bash
  python crop_boxes.py -i input_form.jpg -o output_directory/ --size 128 128
  ```

- **Adjust inner margin from borders (in mm):**
  ```bash
  python crop_boxes.py -i input_form.jpg -o output_directory/ --margin-mm 1.5
  ```

- **Disable automatic paper deskew:**
  ```bash
  python crop_boxes.py -i input_form.jpg -o output_directory/ --no-deskew
  ```

- **Tight-crop directly to character content:**
  ```bash
  python crop_boxes.py -i input_form.jpg -o output_directory/ --crop-to-content
  ```

- **View full CLI options:**
  ```bash
  python crop_boxes.py --help
  ```

---

## 📤 Output Structure

Cropped cells are automatically structured into folders corresponding to each box index:

```text
output_directory/
├── Box 1/
│   ├── document (1)_a.jpg
│   ├── document (2)_a.jpg
│   └── ...
├── Box 2/
│   ├── document (1)_b.jpg
│   ├── document (2)_b.jpg
│   └── ...
├── ...
├── Box 9/
│   ├── document (1)_i.jpg
│   └── ...
└── document (1)_20260901_134657.jpg   # Full-page debug overlay with timestamp
```

- **Cell Filenames:** Original image basename + alphabetical box suffix (`_a` for Box 1, `_b` for Box 2, etc.).
- **Debug Verification:** Annotated full image stamped with date and time for auditing.

---

## 📦 Key Dependencies

| Package | Minimum Version | Description |
|---|---|---|
| `opencv-python` | `>=4.8.0` | Computer vision, edge & line detection, contour analysis, deskewing |
| `flask` | `>=2.0.0` | Web server backend & RESTful processing endpoints |
| `numpy` | `>=1.24.0` | Multidimensional image matrix calculations |
| `Pillow` | `>=10.0.0` | Image format conversion and saving |
| `tkinter` | *Standard Library* | Cross-platform desktop interface |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
