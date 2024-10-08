import numpy as np


def bayer_mosaic(clean_image_content, pixel_order='rggb'):
    corrupt_image_content = clean_image_content.copy()
    image_height, image_width, _ = corrupt_image_content.shape
    mosaic_image_content = np.zeros((image_height, image_width))
    if pixel_order == 'rggb':
        mosaic_image_content[::2,
                             ::2] = corrupt_image_content[::2, ::2, 2]  # R
        mosaic_image_content[::2,
                             1::2] = corrupt_image_content[::2, 1::2, 1]  # G
        mosaic_image_content[1::2,
                             ::2] = corrupt_image_content[1::2, ::2, 1]  # G
        mosaic_image_content[1::2,
                             1::2] = corrupt_image_content[1::2, 1::2, 0]  # B
    elif pixel_order == 'bggr':
        mosaic_image_content[::2,
                             ::2] = corrupt_image_content[::2, ::2, 0]  # B
        mosaic_image_content[::2,
                             1::2] = corrupt_image_content[::2, 1::2, 1]  # G
        mosaic_image_content[1::2,
                             ::2] = corrupt_image_content[1::2, ::2, 1]  # G
        mosaic_image_content[1::2,
                             1::2] = corrupt_image_content[1::2, 1::2, 2]  # R
    elif pixel_order == 'grbg':
        mosaic_image_content[::2,
                             ::2] = corrupt_image_content[::2, ::2, 1]  # G
        mosaic_image_content[::2,
                             1::2] = corrupt_image_content[::2, 1::2, 2]  # R
        mosaic_image_content[1::2,
                             ::2] = corrupt_image_content[1::2, ::2, 0]  # B
        mosaic_image_content[1::2,
                             1::2] = corrupt_image_content[1::2, 1::2, 1]  # G
    elif pixel_order == 'gbrg':
        mosaic_image_content[::2,
                             ::2] = corrupt_image_content[::2, ::2, 1]  # G
        mosaic_image_content[::2,
                             1::2] = corrupt_image_content[::2, 1::2, 0]  # B
        mosaic_image_content[1::2,
                             ::2] = corrupt_image_content[1::2, ::2, 2]  # R
        mosaic_image_content[1::2,
                             1::2] = corrupt_image_content[1::2, 1::2, 1]  # G
    return np.expand_dims(mosaic_image_content, axis=2)
