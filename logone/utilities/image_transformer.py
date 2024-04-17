import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTransform

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

def cylindricalWarp(img, K, rev=False):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    # mapx, mapy = B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32)
    # if rev:
    #     Binvx, Binvy = invert_map(B[:,:,0].astype(np.float32).flatten(), B[:,:,1].astype(np.float32).flatten())
    #     return cv2.remap(img_rgba, Binvx.reshape(B.shape[:2]), Binvy.reshape(B.shape[:2]), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    # else:
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

    
def projective_transformation(img, K):
    """
    warp img accroding to projective matrix K
    """
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    X = (K.dot(X.T).T).reshape(h_,w_,-1)

    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...

    return cv2.remap(img_rgba, X[:,:,0].astype(np.float32), X[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def apply_cylindrical_transformation(img):
    h, w = img.shape[:2]
    diag = 500
    K = np.array([[diag,0,w/2],[0,diag,h/2],[0,0,1]]) # mock intrinsics
    return cylindricalWarp(img, K)

def apply_projective_transformation(img):
    K = np.array([])
    return projective_transformation(img, K)

        
def test_transformation(img_file):
    img_filepath = os.path.join(os.getcwd(),'test_images', 'original',img_file)
    with Image.open(img_filepath) as img:
        basename = os.path.basename(img_file)
        save_dir = os.path.join(os.getcwd(), 'test_images', 'transformed')
        
        # Affine transformation
        img_affine = apply_affine_transform(img)
        img_affine.save(os.path.join(save_dir, f'{basename}_affine.jpg'))
        
        # Cylindrical transformation
        cv_img = cv2.imread(img_filepath)
        img_cylindrical = apply_cylindrical_transformation(cv_img)
        cv2.imwrite(os.path.join(save_dir, f'{basename}_cylindrical.jpg'), img_cylindrical)

        # projective transformation
        cv_img = cv2.imread(img_filepath)
        img_projective = apply_projective_transformation(cv_img)
        cv2.imwrite(os.path.join(save_dir, f'{basename}_projective.jpg'), img_projective)
        
        # Projective transformation
        # img_projective = apply_projective_transform(img)
        # img_projective.save(os.path.join(save_dir, f'{basename}_projective.jpg'))
        
        # Rotation
        img_rotated = apply_rotation(img)
        img_rotated.save(os.path.join(save_dir,f'{basename}_rotated.jpg'))
        
        # Shear transformation
        img_sheared = apply_shear(img)
        img_sheared.save(os.path.join(save_dir,f'{basename}_sheared.jpg'))

# # Processing all images in the original_images directory
# for filename in os.listdir('original_images'):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         transform_and_save(os.path.join('original_images', filename))
if __name__ == "__main__":
    test_transformation('lego.jpg')