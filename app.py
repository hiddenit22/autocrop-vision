import sys
sys.stdout.reconfigure(line_buffering=True)
"""
Web App Crop Otomatis Box Gambar — Flask
Akses di: http://localhost:5000
Mendukung: Template Potong (TP01 dll.) & Batch Processing (banyak file sekaligus).
"""

import os
import uuid
import zipfile
import json
import shutil
from pathlib import Path
from flask import (Flask, render_template, request,
                   jsonify, send_file, abort)
import sys

sys.path.insert(0, os.path.dirname(__file__))
from crop_boxes import process_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # max 128MB untuk batch upload

UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
OUTPUT_FOLDER = Path(__file__).parent / 'output_crops'
TEMPLATES_FILE = Path(__file__).parent / 'templates.json'

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


def natural_sort_key(s):
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────
#  ENDPOINTS TEMPLATE
# ─────────────────────────────────────────────

@app.route('/templates', methods=['GET'])
def get_templates():
    if not TEMPLATES_FILE.exists():
        return jsonify({})
    try:
        with open(TEMPLATES_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/templates', methods=['POST'])
def save_template():
    req = request.get_json()
    if not req or 'name' not in req or 'settings' not in req:
        return jsonify({'error': 'Parameter tidak lengkap'}), 400

    name = req['name'].strip()
    settings = req['settings']

    data = {}
    if TEMPLATES_FILE.exists():
        try:
            with open(TEMPLATES_FILE, 'r') as f:
                data = json.load(f)
        except:
            pass

    data[name] = settings

    try:
        with open(TEMPLATES_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'success': True, 'templates': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  ENDPOINT CROP (Batch & Single)
# ─────────────────────────────────────────────

@app.route('/crop', methods=['POST'])
def crop():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    files = request.files.getlist('file')
    valid_files = [f for f in files if f.filename and allowed(f.filename)]
    valid_files.sort(key=lambda f: natural_sort_key(f.filename))
    if not valid_files:
        return jsonify({'error': 'Tidak ada file dengan format gambar yang didukung'}), 400

    method        = request.form.get('method', 'auto')
    margin_mm     = float(request.form.get('margin_mm', 3.0))
    out_dpi       = int(request.form.get('out_dpi', 100))
    resize        = request.form.get('resize', 'true') == 'true'
    size_h        = int(request.form.get('size_h', 128))
    size_w        = int(request.form.get('size_w', 128))
    min_area      = int(request.form.get('min_area', 5000))
    only_squares  = request.form.get('only_squares', 'false') == 'true'
    only_red      = request.form.get('only_red', 'false') == 'true'
    clean_red     = request.form.get('clean_red', 'false') == 'true'
    exclude_empty      = request.form.get('exclude_empty', 'false') == 'true'
    exclude_top        = float(request.form.get('exclude_top_pct', 0.22))
    group_by_box       = request.form.get('group_by_box', 'false') == 'true'
    custom_output_dir  = request.form.get('custom_output_dir', '').strip()
    deskew             = request.form.get('deskew', 'true') == 'true'
    crop_to_content    = request.form.get('crop_to_content', 'false') == 'true'
    
    manual_boxes_str = request.form.get('manual_boxes', '')
    manual_boxes = []
    if manual_boxes_str:
        import json
        try:
            manual_boxes = json.loads(manual_boxes_str)
        except Exception as e:
            print(f"[WARN] Gagal parse manual_boxes: {e}")

    uid = uuid.uuid4().hex[:10]
    out_dir = OUTPUT_FOLDER / uid
    out_dir.mkdir(exist_ok=True)

    results = []
    total_cropped = 0

    for file in valid_files:
        stem = Path(file.filename).stem
        # Simpan file upload
        in_path = UPLOAD_FOLDER / f"{uid}_{file.filename}"
        file.save(str(in_path))

        try:
            saved = process_image(
                image_path=str(in_path),
                output_dir=str(out_dir),
                original_filename=file.filename,
                method=method,
                margin_mm=margin_mm,
                resize_output=resize,
                target_size=(size_h, size_w),
                min_area=min_area,
                save_debug=True,
                only_squares=only_squares,
                only_red=only_red,
                clean_red=clean_red,
                exclude_top_pct=exclude_top,
                exclude_empty=exclude_empty,
                group_by_box=group_by_box,
                deskew=deskew,
                crop_to_content=crop_to_content,
                manual_boxes=manual_boxes,
                out_dpi=out_dpi
            )

            if saved:
                saved_names = {Path(s).name.lower() for s in saved}
                # Kumpulkan semua file hasil crop dan file utuh (debug) secara rekursif
                for p in sorted(out_dir.rglob('*')):
                    if p.is_file() and p.suffix.lower() in ALLOWED_EXT and p.name != 'all_crops.zip':
                        rel_path = p.relative_to(out_dir).as_posix()
                        is_crop = p.name.lower() in saved_names
                        is_debug = not is_crop
                        display_name = f"File Utuh ({p.name})" if is_debug else (f"{p.parent.name} ({p.name})" if p.parent != out_dir else p.name)
                        
                        if not any(r['url'] == f'/result/{uid}/{rel_path}' for r in results):
                            results.append({
                                'image_name': file.filename,
                                'name': display_name,
                                'url': f'/result/{uid}/{rel_path}',
                                'is_debug': is_debug
                            })
                total_cropped += len([s for s in saved if s])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Gagal memproses {file.filename}: {e}")

    if total_cropped == 0:
        return jsonify({'error': 'Tidak ada box yang berhasil di-crop dari semua file. Coba periksa pengaturan Anda.'}), 422

    # Salin hasil crop ke folder eksternal lokal jika diisi
    if custom_output_dir:
        try:
            custom_path = Path(custom_output_dir)
            custom_path.mkdir(parents=True, exist_ok=True)
            for item in out_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(str(item), str(custom_path / item.name), dirs_exist_ok=True)
                elif item.is_file() and item.suffix.lower() in ALLOWED_EXT and item.suffix != '.zip':
                    shutil.copy2(str(item), str(custom_path / item.name))
            print(f"[OK] Berhasil menyalin hasil ke folder lokal kustom: {custom_output_dir}")
        except Exception as e:
            print(f"[ERROR] Gagal menyalin ke folder lokal kustom: {e}")

    return jsonify({
        'status': 'success',
        'total': total_cropped,
        'uid': uid,
        'files': results
    })


@app.route('/result/<uid>/<path:filepath>')
def serve_result(uid, filepath):
    path = OUTPUT_FOLDER / uid / filepath
    if not path.exists():
        abort(404)
    return send_file(str(path))


@app.route('/download/<uid>')
def download_zip(uid):
    out_dir = OUTPUT_FOLDER / uid
    if not out_dir.exists():
        abort(404)

    zip_path = out_dir / 'all_crops.zip'
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        # Loop semua file hasil secara rekursif dan masukkan ke ZIP dengan struktur foldernya
        for f in sorted(out_dir.rglob('*')):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT and f.name != 'all_crops.zip':
                rel_p = f.relative_to(out_dir)
                zf.write(str(f), arcname=str(rel_p))

    return send_file(str(zip_path), as_attachment=True,
                     download_name='crops_terkelompok.zip')


@app.route('/get_local_images', methods=['POST'])
def get_local_images():
    folder = request.form.get('local_folder', '').strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({'error': 'Folder tidak valid atau tidak ditemukan'}), 400
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            images.append(os.path.join(folder, f))
            
    images.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    return jsonify({'status': 'success', 'images': images})

@app.route('/crop_exact', methods=['POST'])
def crop_exact():
    # Mirip dengan /crop tapi hanya memotong berdasarkan array exact_boxes
    if 'file' not in request.files and 'file_path' not in request.form:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    exact_boxes_str = request.form.get('exact_boxes', '')
    exact_boxes = []
    if exact_boxes_str:
        try:
            exact_boxes = json.loads(exact_boxes_str)
        except Exception:
            pass
            
    margin_mm     = float(request.form.get('margin_mm', 3.0))
    out_dpi       = int(request.form.get('out_dpi', 100))
    resize        = request.form.get('resize', 'true') == 'true'
    size_h        = int(request.form.get('size_h', 128))
    size_w        = int(request.form.get('size_w', 128))
    min_area      = int(request.form.get('min_area', 5000))
    only_squares  = request.form.get('only_squares', 'false') == 'true'
    only_red      = request.form.get('only_red', 'false') == 'true'
    clean_red     = request.form.get('clean_red', 'false') == 'true'
    exclude_empty      = request.form.get('exclude_empty', 'false') == 'true'
    exclude_top        = float(request.form.get('exclude_top_pct', 0.22))
    group_by_box       = request.form.get('group_by_box', 'false') == 'true'
    custom_output_dir  = request.form.get('custom_output_dir', '').strip()
    deskew             = request.form.get('deskew', 'true') == 'true'
    crop_to_content    = request.form.get('crop_to_content', 'false') == 'true'

    uid = uuid.uuid4().hex[:10]
    out_dir = OUTPUT_FOLDER / uid
    out_dir.mkdir(exist_ok=True)
    
    img_path = None
    tmp_path = None
    is_local = False
    file_name = ""
    
    if 'file_path' in request.form:
        img_path = request.form['file_path']
        is_local = True
        file_name = Path(img_path).name
    elif 'file' in request.files:
        file = request.files['file']
        file_name = file.filename
        img_path = str(UPLOAD_FOLDER / f"{uid}_{file.filename}")
        file.save(img_path)

    try:
        saved = process_image(
            image_path=img_path,
            output_dir=str(out_dir),
            original_filename=file_name,
            method='exact',
            margin_mm=margin_mm,
            resize_output=resize,
            target_size=(size_h, size_w),
            min_area=min_area,
            save_debug=True,
            only_squares=only_squares,
            only_red=only_red,
            clean_red=clean_red,
            exclude_top_pct=exclude_top,
            exclude_empty=exclude_empty,
            group_by_box=group_by_box,
            deskew=deskew,
            crop_to_content=crop_to_content,
            exact_boxes=exact_boxes,
            out_dpi=out_dpi
        )
        
        # Salin ke custom output dir jika disetel (menimpa/menghapus file lama dari gambar ini)
        if custom_output_dir and Path(custom_output_dir).exists():
            final_dir = Path(custom_output_dir)
            file_stem = Path(file_name).stem
            
            # Hapus file hasil crop lama dari gambar ini di folder Box 1 s/d Box N
            for box_dir in final_dir.glob("Box *"):
                if box_dir.is_dir():
                    for old_crop in box_dir.glob(f"{file_stem}_*.*"):
                        try:
                            old_crop.unlink()
                        except Exception as e:
                            print(f"[WARN] Gagal menghapus crop lama {old_crop}: {e}")
                            
            # Hapus file utuh/debug lama dari gambar ini
            for old_debug in final_dir.glob(f"{file_stem}_*.*"):
                if old_debug.is_file():
                    try:
                        old_debug.unlink()
                    except Exception as e:
                        print(f"[WARN] Gagal menghapus file debug lama {old_debug}: {e}")

            # Salin hasil crop & debug yang baru
            for f in out_dir.iterdir():
                dest = final_dir / f.name
                if f.is_dir():
                    shutil.copytree(str(f), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(f), str(dest))

        saved_names = {Path(s).name.lower() for s in saved}
        results = []
        for p in sorted(out_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT and p.name != 'all_crops.zip':
                rel_path = p.relative_to(out_dir).as_posix()
                is_crop = p.name.lower() in saved_names
                is_debug = not is_crop
                display_name = f"File Utuh ({p.name})" if is_debug else (f"{p.parent.name} ({p.name})" if p.parent != out_dir else p.name)
                results.append({
                    'image_name': file_name,
                    'file_path': img_path if is_local else '',
                    'name': display_name,
                    'url': f'/result/{uid}/{rel_path}',
                    'is_debug': is_debug
                })

        return jsonify({
            'status': 'success',
            'total': len([s for s in saved if s]),
            'uid': uid,
            'files': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_crop_boxes', methods=['POST'])
def generate_crop_boxes():
    img_path = None
    tmp_path = None
    if 'file_path' in request.form:
        img_path = request.form['file_path']
    elif 'file' in request.files:
        file = request.files['file']
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(tmp.name)
        img_path = tmp.name
        tmp_path = tmp.name
        tmp.close()
        
    if not img_path or not os.path.isfile(img_path):
        return jsonify({'error': 'No valid file provided'}), 400

    margin_mm = float(request.form.get('margin_mm', 2.0))
    deskew = request.form.get('deskew', 'true') == 'true'
    manual_boxes_str = request.form.get('manual_boxes', '')
    manual_boxes = []
    if manual_boxes_str:
        import json
        try:
            manual_boxes = json.loads(manual_boxes_str)
        except Exception:
            pass

    method = 'manual' if manual_boxes else 'auto'

    try:
        boxes = process_image(
            image_path=img_path,
            output_dir=str(OUTPUT_FOLDER),
            method=method,
            margin_mm=margin_mm,
            deskew=deskew,
            manual_boxes=manual_boxes,
            return_boxes_only=True
        )
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({'status': 'success', 'boxes': boxes})
    except Exception as e:
        import traceback
        traceback.print_exc()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/preview_auto_boxes', methods=['POST'])
def preview_auto_boxes():
    img_path = None
    tmp_path = None
    if 'file_path' in request.form:
        img_path = request.form['file_path']
    elif 'file' in request.files:
        file = request.files['file']
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(tmp.name)
        img_path = tmp.name
        tmp_path = tmp.name
        tmp.close()
        
    if not img_path or not os.path.isfile(img_path):
        return jsonify({'error': 'No valid file provided'}), 400

    try:
        from crop_boxes import detect_grid, crop_cells_from_grid, preprocess_image
        import cv2
        image = cv2.imread(img_path)
        if image is None:
            return jsonify({'error': 'Cannot read image'}), 400
            
        h, w = image.shape[:2]
        gray, thresh = preprocess_image(image)
        rows, cols, h_thick, v_thick, horiz_m, vert_m = detect_grid(gray, thresh, image.shape)
        
        cells_nested = crop_cells_from_grid(image, gray, rows, cols, h_thick, v_thick, 4, 4)
        cells_flat = [c for r in cells_nested for c in r]
        
        boxes = []
        for cell in cells_flat:
            x1, y1, x2, y2 = cell['bbox']
            boxes.append({
                'x': x1 / w,
                'y': y1 / h,
                'w': (x2 - x1) / w,
                'h': (y2 - y1) / h
            })
            
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return jsonify({'status': 'success', 'boxes': boxes})
    except Exception as e:
        import traceback
        traceback.print_exc()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/crop_local_file', methods=['POST'])
def crop_local_file():
    file_path = request.form.get('file_path', '').strip()
    if not file_path or not os.path.isfile(file_path):
        return jsonify({'error': 'File tidak ditemukan'}), 400

    method        = request.form.get('method', 'auto')
    margin_mm     = float(request.form.get('margin_mm', 3.0))
    out_dpi       = int(request.form.get('out_dpi', 100))
    resize        = request.form.get('resize', 'true') == 'true'
    size_h        = int(request.form.get('size_h', 128))
    size_w        = int(request.form.get('size_w', 128))
    min_area      = int(request.form.get('min_area', 5000))
    only_squares  = request.form.get('only_squares', 'false') == 'true'
    method        = request.form.get('method', 'auto')
    margin_mm     = float(request.form.get('margin_mm', 3.0))
    out_dpi       = int(request.form.get('out_dpi', 100))
    resize        = request.form.get('resize', 'true') == 'true'
    size_h        = int(request.form.get('size_h', 128))
    size_w        = int(request.form.get('size_w', 128))
    min_area      = int(request.form.get('min_area', 5000))
    only_squares  = request.form.get('only_squares', 'false') == 'true'
    only_red      = request.form.get('only_red', 'false') == 'true'
    clean_red     = request.form.get('clean_red', 'false') == 'true'
    exclude_empty      = request.form.get('exclude_empty', 'false') == 'true'
    exclude_top        = float(request.form.get('exclude_top_pct', 0.22))
    group_by_box       = request.form.get('group_by_box', 'false') == 'true'
    custom_output_dir  = request.form.get('custom_output_dir', '').strip()
    deskew             = request.form.get('deskew', 'true') == 'true'
    crop_to_content    = request.form.get('crop_to_content', 'false') == 'true'
    
    manual_boxes_str = request.form.get('manual_boxes', '')
    manual_boxes = []
    if manual_boxes_str:
        import json
        try:
            manual_boxes = json.loads(manual_boxes_str)
        except Exception:
            pass

    uid = uuid.uuid4().hex[:10]
    out_dir = OUTPUT_FOLDER / uid
    out_dir.mkdir(exist_ok=True)

    stem = Path(file_path).stem
    try:
        saved_files = process_image(
            file_path,
            str(out_dir),
            original_filename=Path(file_path).name,
            method=method,
            margin_mm=margin_mm,
            resize_output=resize,
            target_size=(size_h, size_w),
            min_area=min_area,
            save_debug=True,
            only_squares=only_squares,
            only_red=only_red,
            clean_red=clean_red,
            exclude_top_pct=exclude_top,
            exclude_bottom_pct=0.05,
            exclude_empty=exclude_empty,
            group_by_box=group_by_box,
            deskew=deskew,
            crop_to_content=crop_to_content,
            out_dpi=out_dpi,
            manual_boxes=manual_boxes
        )
        
        # Salin ke custom output dir jika disetel
        if custom_output_dir and Path(custom_output_dir).exists():
            final_dir = Path(custom_output_dir)
            for f in out_dir.iterdir():
                dest = final_dir / f.name
                if f.is_dir():
                    shutil.copytree(str(f), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(f), str(dest))

        saved_names = {Path(s).name.lower() for s in saved_files}
        results = []
        for p in sorted(out_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT and p.name != 'all_crops.zip':
                rel_path = p.relative_to(out_dir).as_posix()
                is_crop = p.name.lower() in saved_names
                is_debug = not is_crop
                display_name = f"File Utuh ({p.name})" if is_debug else (f"{p.parent.name} ({p.name})" if p.parent != out_dir else p.name)
                
                if not any(r['url'] == f'/result/{uid}/{rel_path}' for r in results):
                    results.append({
                        'image_name': Path(file_path).name,
                        'file_path': file_path,
                        'name': display_name,
                        'url': f'/result/{uid}/{rel_path}',
                        'is_debug': is_debug
                    })

        total_cropped = len([s for s in saved_files if s])
        return jsonify({
            'status': 'success',
            'total': total_cropped,
            'uid': uid,
            'files': results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/serve_local_image')
def serve_local_image():
    path = request.args.get('path', '')
    if not path or not os.path.isfile(path):
        return "Not found", 404
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    from flask import send_from_directory
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  [WEB] Web App Crop Gambar (Batch + Template)")
    print("  Buka browser: http://localhost:5050")
    print("="*45 + "\n")

    app.run(debug=False, port=5050, host='0.0.0.0')
