import cv2
import numpy as np

# 1 Padding
def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )
    cv2.imwrite('padded_image.png', padded_image)
    return padded_image

# 2 Cropping
def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite('cropped_image.png', cropped_image)
    return cropped_image

# 3 Resize
def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite('resized_image.png', resized_image)
    return resized_image

# 4 Manual Copy (pixel-by-pixel)
def copy(image, emptyPictureArray):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite('copied_image.png', emptyPictureArray)
    return emptyPictureArray

# 5 Grayscale
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grayscale_image.png', gray)
    return gray

# 6 HSV
def hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv_image.png', hsv_img)
    return hsv_img

# 7 Hue shift
def hue_shifted(image, emptyPictureArray, hue):
    # Convert to a larger integer type to safely add hue
    shifted = image.astype(np.int16) + int(hue)

    # Clamp values to the valid 0–255 range
    shifted = np.clip(shifted, 0, 255)

    # Convert back to uint8
    shifted = shifted.astype(np.uint8)

    # Save and return
    cv2.imwrite('hue_shifted_image.png', shifted)
    return shifted

# 8 Smoothing / Blur
def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite('smoothed_image.png', blurred)
    return blurred

# 9 Rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("rotation_angle must be 90 or 180")
    cv2.imwrite(f'rotated_{rotation_angle}.png', rotated)
    return rotated


def main():
    # Load the image
    img1 = cv2.imread('lena-1.png')
    if img1 is None:
        print("Error: lena-1.png not found")
        return

    # Padding
    padded = padding(img1, 100)

    # Cropping (80px from top/left, 130px from bottom/right)
    h, w = img1.shape[:2]
    cropped = crop(img1, 80, w - 130, 80, h - 130)

    # Resize to 200x200
    resized = resize(img1, 200, 200)

    # Manual copy
    empty_array = np.zeros_like(img1, dtype=np.uint8)
    copied = copy(img1, empty_array)

    # Grayscale
    gray = grayscale(img1)

    # HSV
    hsv_img = hsv(img1)

    # Hue shift by +50
    empty_shift = np.zeros_like(img1, dtype=np.uint8)
    hue_img = hue_shifted(img1, empty_shift, 50)

    # Smoothing
    blurred = smoothing(img1)

    # Rotations
    rot_90 = rotation(img1, 90)
    rot_180 = rotation(img1, 180)


if __name__ == '__main__':
    main()
