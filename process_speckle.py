import os, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io, img_as_float, restoration, exposure, util
import cv2

USE_OPENCV_NLM = False       # True = faster (OpenCV), False = skimage NLM
PATCH_SIZE = 5
PATCH_DISTANCE = 6
H_FACTOR = 1.15              # skimage multiplier for sigma
OPENCV_H = 10                # OpenCV denoising strength
CLAHE_CLIP = 0.01
CLAHE_KERNEL = None

# helper functions
def find_all_images(root='/content/drive/MyDrive'):
    exts = ('*.png','*.jpg','*.jpeg','*.tif','*.bmp')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, '**', e), recursive=True)
    return sorted(list(dict.fromkeys(files)))

def grayscale_normalize(img):
    imgf = img_as_float(img)
    mn, mx = float(imgf.min()), float(imgf.max())
    if mx - mn < 1e-12:
        return np.zeros_like(imgf)
    return (imgf - mn) / (mx - mn)

def nlm_skimage(img, patch_size=5, patch_distance=6, h_factor=1.15):
    sigma_est = max(1e-12, float(np.mean(restoration.estimate_sigma(img, channel_axis=None))))
    h = h_factor * sigma_est
    return restoration.denoise_nl_means(img, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True, channel_axis=None)

def nlm_opencv(img, h=10):
    u8 = (np.clip(img,0,1)*255).astype('uint8')
    den = cv2.fastNlMeansDenoising(u8, None, h, 7, 21)
    return den.astype('float32')/255.0

def apply_clahe(img, clip_limit=0.01, kernel_size=None):
    return exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)

def save_uint8(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.imsave(path, util.img_as_ubyte(np.clip(img,0,1)))

def show_pair(speckle, processed, title=""):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(speckle, cmap='gray'); plt.title('Speckle (input)'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(processed, cmap='gray'); plt.title('Processed (NLM+CLAHE)'); plt.axis('off')
    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    ALL_IMAGES = find_all_images('/content/drive/MyDrive')
    print("Total images found under MyDrive:", len(ALL_IMAGES))
    
    if len(ALL_IMAGES) == 0:
        raise SystemExit("No image files found in /content/drive/MyDrive. Upload images to Drive or use files.upload() in Colab.")
    
    # pick targets: filenames containing 'speckle' (case-insensitive). If none found -> process all images.
    speckle_candidates = [p for p in ALL_IMAGES if 'speckle' in Path(p).stem.lower()]
    if not speckle_candidates:
        print("No filenames with 'speckle' found — will process ALL images found.")
        speckle_candidates = ALL_IMAGES
    else:
        print("Found speckle images:", [os.path.basename(p) for p in speckle_candidates])
    
    # process each speckle candidate
    for p in tqdm(speckle_candidates, desc="Processing speckle images"):
        try:
            img = img_as_float(io.imread(p, as_gray=True))
        except Exception as e:
            print("Failed to read", p, ":", e)
            continue

        # 1) grayscale normalization
        speckle_norm = grayscale_normalize(img)

        # 2) NLM denoising
        try:
            if USE_OPENCV_NLM:
                den = nlm_opencv(speckle_norm, h=OPENCV_H)
            else:
                den = nlm_skimage(speckle_norm, patch_size=PATCH_SIZE, patch_distance=PATCH_DISTANCE, h_factor=H_FACTOR)
        except Exception as e:
            print("NLM failed for", p, ":", e)
            den = speckle_norm.copy()

        # 3) CLAHE
        proc = apply_clahe(den, clip_limit=CLAHE_CLIP, kernel_size=CLAHE_KERNEL)

        # 4) save outputs in folder next to the original file
        outdir = os.path.join(os.path.dirname(p), 'output_speckle')
        base = Path(p).stem
        save_uint8(os.path.join(outdir, base + '_speckle_norm.png'), speckle_norm)
        save_uint8(os.path.join(outdir, base + '_den.png'), den)
        save_uint8(os.path.join(outdir, base + '_proc.png'), proc)

        # 5) display speckle input and processed output
        title = f"{os.path.basename(p)}  → saved to {outdir}"
        show_pair(speckle_norm, proc, title=title)
    
    print("Done. Processed outputs are in each image's 'output_speckle' folder in Drive.")
