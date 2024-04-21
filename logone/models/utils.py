import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def bilinear_inverse(p, vertices, numiter=4):
    """
    Compute the inverse of the bilinear map from the unit square
    [(0,0), (1,0), (1,1), (0,1)]
    to the quadrilateral vertices = [p0, p1, p2, p4]

    Parameters:
    ----------
    p: array of shape (2, ...)
        Points on which the inverse transforms are applied.
    vertices: array of shape (4, 2, ...)
        Coordinates of the vertices mapped to the unit square corners
    numiter:
        Number of Newton interations

    Returns:
    --------
    s: array of shape (2, ...)
        Mapped points.

    This is a (more general) python implementation of the matlab implementation 
    suggested in https://stackoverflow.com/a/18332009/1560876
    """

    p = np.asarray(p)
    v = np.asarray(vertices)
    sh = p.shape[1:]
    if v.ndim == 2:
        v = np.expand_dims(v, axis=tuple(range(2, 2 + len(sh))))

    # Start in the center
    s = .5 * np.ones((2,) + sh)
    s0, s1 = s
    for k in range(numiter):
        # Residual
        r = v[0] * (1 - s0) * (1 - s1) + v[1] * s0 * (1 - s1) + v[2] * s0 * s1 + v[3] * (1 - s0) * s1 - p

        # Jacobian
        J11 = -v[0, 0] * (1 - s1) + v[1, 0] * (1 - s1) + v[2, 0] * s1 - v[3, 0] * s1
        J21 = -v[0, 1] * (1 - s1) + v[1, 1] * (1 - s1) + v[2, 1] * s1 - v[3, 1] * s1
        J12 = -v[0, 0] * (1 - s0) - v[1, 0] * s0 + v[2, 0] * s0 + v[3, 0] * (1 - s0)
        J22 = -v[0, 1] * (1 - s0) - v[1, 1] * s0 + v[2, 1] * s0 + v[3, 1] * (1 - s0)

        inv_detJ = 1. / (J11 * J22 - J12 * J21)

        s0 -= inv_detJ * (J22 * r[0] - J12 * r[1])
        s1 -= inv_detJ * (-J21 * r[0] + J11 * r[1])

    return s

def invert_map(xmap, ymap, diagnostics=False):
    """
    Generate the inverse of deformation map defined by (xmap, ymap) using inverse bilinear interpolation.
    """

    # Generate quadrilaterals from mapped grid points.
    quads = np.array([[ymap[:-1, :-1], xmap[:-1, :-1]],
                      [ymap[1:, :-1], xmap[1:, :-1]],
                      [ymap[1:, 1:], xmap[1:, 1:]],
                      [ymap[:-1, 1:], xmap[:-1, 1:]]])

    # Range of indices possibly within each quadrilateral
    x0 = np.floor(quads[:, 1, ...].min(axis=0)).astype(int)
    x1 = np.ceil(quads[:, 1, ...].max(axis=0)).astype(int)
    y0 = np.floor(quads[:, 0, ...].min(axis=0)).astype(int)
    y1 = np.ceil(quads[:, 0, ...].max(axis=0)).astype(int)

    # Quad indices
    i0, j0 = np.indices(x0.shape)

    # Offset of destination map
    x0_offset = x0.min()
    y0_offset = y0.min()

    # Index range in x and y (per quad)
    xN = x1 - x0 + 1
    yN = y1 - y0 + 1

    # Shape of destination array
    sh_dest = (1 + x1.max() - x0_offset, 1 + y1.max() - y0_offset)

    # Coordinates of destination array
    yy_dest, xx_dest = np.indices(sh_dest)

    xmap1 = np.zeros(sh_dest)
    ymap1 = np.zeros(sh_dest)
    TN = np.zeros(sh_dest, dtype=int)

    # Smallish number to avoid missing point lying on edges
    epsilon = .01

    # Loop through indices possibly within quads
    for ix in range(xN.max()):
        for iy in range(yN.max()):
            # Work only with quads whose bounding box contain indices
            valid = (xN > ix) * (yN > iy)

            # Local points to check
            p = np.array([y0[valid] + ix, x0[valid] + iy])

            # Map the position of the point in the quad
            s = bilinear_inverse(p, quads[:, :, valid])

            # s out of unit square means p out of quad
            # Keep some epsilon around to avoid missing edges
            in_quad = np.all((s > -epsilon) * (s < (1 + epsilon)), axis=0)

            # Add found indices
            ii = p[0, in_quad] - y0_offset
            jj = p[1, in_quad] - x0_offset

            ymap1[ii, jj] += i0[valid][in_quad] + s[0][in_quad]
            xmap1[ii, jj] += j0[valid][in_quad] + s[1][in_quad]

            # Increment count
            TN[ii, jj] += 1

    ymap1 /= TN + (TN == 0)
    xmap1 /= TN + (TN == 0)

    if diagnostics:
        diag = {'x_offset': x0_offset,
                'y_offset': y0_offset,
                'mask': TN > 0}
        return xmap1, ymap1, diag
    else:
        return xmap1, ymap1

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
    mapx, mapy = B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32)
    print(mapx.shape)
    print(mapy.shape)
    # if rev:
    #     Binvx, Binvy = invert_map(B[:,:,0].astype(np.float32).flatten(), B[:,:,1].astype(np.float32).flatten())
    #     return cv2.remap(img_rgba, Binvx.reshape(B.shape[:2]), Binvy.reshape(B.shape[:2]), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    # else:
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def normalize_logo_256(img):
    h,w = img.shape[:2]
    lg_res = max(img.shape[:2])
    sq_img = np.ones((lg_res,lg_res,3), np.uint8) * 255
    h_offset = (lg_res - h)//2
    w_offset = (lg_res - w)//2
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
    if img.shape[2] == 4:
        # print(img[:,:,3])
        img[img[:,:,3]==0] = np.array([255,255,255,255])
    sq_img[h_offset:h_offset+h, w_offset:w_offset+w, :] = img[:,:,:3]
    resize_sq_img = np.ones((256,256,3), np.uint8) * 255
    resize_sq_img = cv2.resize(sq_img, resize_sq_img.shape[:2], resize_sq_img, 0, 0)
    return resize_sq_img


def place_transformed_logo(img, res=(1080,1920), offset=(0,0)):
    """
    img (numpy array): image to transform
    res (tuple of ints): (height, width)
    offset (tuples of floats): (x,y) offset of transformed logo placement
    """
    height, width = res
    img_new = np.zeros((height,width,3), np.uint8)
    if height > offset[0]+img.shape[0]:
        new_img_height = height - offset[0]
        new_img_width = width - offset[1]
        img_new[offset[0]:, offset[1]:] = img[:new_img_height, :new_img_width]
    else:
        img_new[offset[0]:offset[0]+img.shape[0], offset[1]:offset[1]+img.shape[1]] = img
    return img_new

def make_border(img, value=255, thickness=100):
    """
    value - give all three channels this valeu
    """
    img_border = np.ones((img.shape[0]+thickness*2, img.shape[1]+thickness*2, img.shape[2]), np.uint8) * value
    img_border[thickness:img.shape[0]+thickness, thickness:img.shape[1]+thickness] = img
    return img_border

def find_bounding_box(img):
    """
    find bounding box around logo given white background
    """
    logo_mask = img < 255
    logo_mask = np.any(logo_mask, axis=-1)
    logo_idxs = logo_mask.nonzero()
    minx = np.min(logo_idxs[0])
    maxx = np.max(logo_idxs[0])
    miny = np.min(logo_idxs[1])
    maxy = np.max(logo_idxs[1])
    logo_mask = np.zeros_like(logo_mask)
    img_crop = img[minx:maxx+1, miny:maxy+1]
    return img_crop

def zoom_to_bounding_box(img):
    """
    zoom image to remove white border
    """
    try: img_crop = find_bounding_box(img)
    except: img_crop = np.ones((256,256,3), np.uint8) * 255
    img_new = normalize_logo_256(img_crop)
    return img_new

if __name__ == "__main__":
    img = cv2.imread(os.path.join(os.getcwd(), 'logone', 'utilities', 'original_256',"adidas_256.jpg"))
    # h, w = img.shape[:2]
    # print('Height | Width \n', h, w)
    # diag = 500
    # K = np.array([[diag,0,w/2],[0,diag,h/2],[0,0,1]]) # mock intrinsics
    # img_cyl = cylindricalWarp(img, K, rev=True)
    # cv2.imwrite("lego_cyl.jpg", img_cyl)
    newimg = zoom_to_bounding_box(img)