import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import glob
import os
import re

def calculate_psnr_ssim(image1_path, image2_path):
    # Read the images
    image1 = img_as_float(io.imread(image1_path))
    image2 = img_as_float(io.imread(image2_path))

    # Ensure the images have the same shape
    #if image1.shape != image2.shape:
        #raise ValueError("Input images must have the same dimensions.")
        
    # Handle (H, W, 3) vs (H, W) mismatch by taking the first channel
    # if the image is 3-dimensional.
    if image1.ndim == 3 and image1.shape[-1] == 3:
        image1 = image1[:, :, 0]
    
    if image2.ndim == 3 and image2.shape[-1] == 3:
        image2 = image2[:, :, 0]

    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError(f"Input images must have the same dimensions. Got {image1.shape} and {image2.shape}")
        
    # Calculate PSNR
    psnr_value = peak_signal_noise_ratio(image1, image2)

    # Calculate SSIM
    ssim_value, _ = structural_similarity(image1, image2, full=True, channel_axis=-1)

    return psnr_value, ssim_value


def calculate_metrics_for_image_pairs(image_pairs):
    psnr_values = []
    ssim_values = []

    for image1_path, image2_path in image_pairs:
        psnr, ssim = calculate_psnr_ssim(image1_path, image2_path)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    psnr_mean = np.mean(psnr_values)
    psnr_std = np.std(psnr_values)
    ssim_mean = np.mean(ssim_values)
    ssim_std = np.std(ssim_values)

    return psnr_mean, psnr_std, ssim_mean, ssim_std

def natural_keys(text):
    '''
    Splits text into a list of strings and integers for natural sorting.
    Example: 'img_2.png' -> ['img_', 2, '.png']
    '''
    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# Example usage:
# Assuming you have pairs of images in two folders: folder1 and folder2
#folder1 = '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_2/real_A'
#folder2 = '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2octo/test_2/fake_B'

# img-2-img turbo hemi
folder1 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_A'
#folder2 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/output/cyclegan_turbo/em_hemi_octo/fid-25001/samples_a2b'
folder2 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/output/cyclegan_turbo/em_hemi_octo/fid-2501/samples_a2b'

#folder1 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/data/em_hemi_octo/test_B'
#folder2 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/output/cyclegan_turbo/em_hemi_octo/fid-25001/samples_b2a'
#folder2 = '/media/samia/DATA/ark/code-experiments/img2img-turbo/output/cyclegan_turbo/em_hemi_octo/fid-2501/samples_b2a'

# Get list of image paths
#image1_paths = sorted(glob.glob(folder1 + '/*.png'))
#image2_paths = sorted(glob.glob(folder2 + '/*.png'))

# Get list of image paths and sort them NATURALLY
# This ensures 2.png comes before 10.png, aligning 0.png with the first input image.
image1_paths = sorted(glob.glob(os.path.join(folder1, '*.png')), key=natural_keys)
image2_paths = sorted(glob.glob(os.path.join(folder2, '*.png')), key=natural_keys)

# Safety Check: Print first few pairs to verify alignment
print("Verifying pairing order (first 3 pairs):")
for i in range(min(3, len(image1_paths))):
    print(f"{os.path.basename(image1_paths[i])}  <-->  {os.path.basename(image2_paths[i])}")
print("-" * 30)


# Ensure both folders have the same number of images
if len(image1_paths) != len(image2_paths):
    raise ValueError("Both folders must contain the same number of images.")

# Pair up the images
image_pairs = list(zip(image1_paths, image2_paths))

# Calculate metrics
psnr_mean, psnr_std, ssim_mean, ssim_std = calculate_metrics_for_image_pairs(image_pairs)

print(f"PSNR Mean: {psnr_mean}, PSNR Std: {psnr_std}")
print(f"SSIM Mean: {ssim_mean}, SSIM Std: {ssim_std}")

# ### real A - hemi fake B hemi like octo check latest
# PSNR Mean: 23.46861581376863, PSNR Std: 0.5815358493430164
# SSIM Mean: 0.9415619978232814, SSIM Std: 0.005334801564290864

#### real B - octo fake A check latest
# PSNR Mean: 20.52315113024133, PSNR Std: 0.22519194812377577
# SSIM Mean: 0.9020819341802198, SSIM Std: 0.0031618992609762833

#### real B - octo fake A  check 2
# PSNR Mean: 19.81296432001419, PSNR Std: 0.35708282882082065
# SSIM Mean: 0.8397114783517791, SSIM Std: 0.005320761340782217

### real A - hemi fake B hemi like octo check 2
# PSNR Mean: 25.199702675305105, PSNR Std: 0.40689263571853906
# SSIM Mean: 0.9153362442451345, SSIM Std: 0.0026599020419724702

### real hemi and fake hemi (samples a2b from img-2-img turbo fid 25000)
#PSNR Mean: 22.140342066529588, PSNR Std: 0.8394054114079815
#SSIM Mean: 0.9454309939808356, SSIM Std: 0.008024617951715303

### real hemi and fake hemi (samples a2b from img-2-img turbo fid 2500)
#PSNR Mean: 24.720026288337547, PSNR Std: 0.5314124879194361
#SSIM Mean: 0.8116623733544847, SSIM Std: 0.013844743786776165

### real octo and fake octo (samples b2a from img-2-img turbo fid 25000)
#PSNR Mean: 19.481433357727976, PSNR Std: 1.8582148507061602
#SSIM Mean: 0.9129184176212919, SSIM Std: 0.014275660630169577

### real octo and fake octo (samples b2a from img-2-img turbo fid 2500)
#PSNR Mean: 18.607937234301986, PSNR Std: 0.8967129430070389
#SSIM Mean: 0.781627504161954, SSIM Std: 0.060239873668327525


