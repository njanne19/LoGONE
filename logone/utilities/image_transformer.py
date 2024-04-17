import os
import numpy as np
from PIL import Image, ImageOps, ImageTransform

# Ensure the output directory exists
if not os.path.exists('transformed_images'):
    os.makedirs('transformed_images')

def apply_affine_transform(image):
    # Applying an example affine transformation (translation)
    return image.transform(image.size, ImageTransform.AffineTransform((1, 0, 50, 0, 1, 50)))

def apply_cylindrical_transform(image):
    # Example cylindrical transform using warp
    width, height = image.size
    def map_cylindrical(x, y):
        return (
            width * (np.arctan((x - width / 2) / width * 2) / np.pi + 0.5),
            height * ((y - height / 2) / (width / 2) * np.sqrt((x - width / 2)**2 + width**2) + 0.5)
        )
    return image.transform(image.size, ImageTransform.QuadTransform(map_cylindrical))

def apply_projective_transform(image):
    # Projective (homography) transformation
    width, height = image.size
    coeffs = ImageTransform.find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [(0, height * 0.1), (width, 0), (width * 0.9, height), (width * 0.1, height * 0.9)]
    )
    return image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def apply_rotation(image):
    # Rotate the image by 45 degrees
    return image.rotate(45)

def apply_shear(image):
    # Shearing the image
    return image.transform(image.size, ImageTransform.AffineTransform((1, 0.5, 0, 0.1, 1, 0)))

def transform_and_save(image_path):
    with Image.open(image_path) as img:
        basename = os.path.basename(image_path)
        
        # Affine transformation
        img_affine = apply_affine_transform(img)
        img_affine.save(f'transformed_images/{basename}_affine.jpg')
        
        # Cylindrical transformation
        # img_cylindrical = apply_cylindrical_transform(img)
        # img_cylindrical.save(f'transformed_images/{basename}_cylindrical.jpg')
        
        # Projective transformation
        # img_projective = apply_projective_transform(img)
        # img_projective.save(f'transformed_images/{basename}_projective.jpg')
        
        # Rotation
        img_rotated = apply_rotation(img)
        img_rotated.save(f'transformed_images/{basename}_rotated.jpg')
        
        # Shear transformation
        img_sheared = apply_shear(img)
        img_sheared.save(f'transformed_images/{basename}_sheared.jpg')

# # Processing all images in the original_images directory
# for filename in os.listdir('original_images'):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         transform_and_save(os.path.join('original_images', filename))
if __name__ == "__main__":
    transform_and_save('lego.jpg')