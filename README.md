**speckle-noise-reduction-nlm-clahe**
Python-based pipeline for speckle noise reduction in grayscale biomedical images using Non-Local Means denoising and CLAHE contrast enhancement.

**Overview**
This project implements a complete image-processing pipeline for reducing speckle noise
in grayscale images commonly observed in biomedical imaging (e.g., ultrasound, OCT).

The pipeline combines Non-Local Means (NLM) denoising with adaptive contrast enhancement
using CLAHE to improve visual clarity while preserving structural details.

**Processing Pipeline**
1. Grayscale normalization
2. Speckle noise reduction using:
   - Non-Local Means (scikit-image)
   - Optional fast NLM using OpenCV
3. Contrast enhancement using CLAHE
4. Batch processing with automatic output saving
5. Side-by-side visualization of input vs processed images

**Tech Stack**
- Python
- NumPy
- OpenCV
- scikit-image
- Matplotlib
- tqdm

**How to Run**
```bash
pip install -r requirements.txt
python process_speckle.py
```
**Output**
For each input image, the following are saved:
- Normalized grayscale image
- Denoised image
- Final processed image (NLM + CLAHE)
Outputs are stored in an output_speckle/ folder alongside the original image.

**Use Cases**
- Biomedical imaging (ultrasound, OCT)
- Speckle-affected grayscale images
- Low-SNR image enhancement
