import cv2
import numpy as np


# Preprocessing the image for use
def preprocessed_image(image, kernel_size=(3,3), sigma=0):
    # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, kernel_size, sigma)

    return img_blur


# 1 Sobel edge detection
def sobel_edge_detection(image):
    sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)

    cv2.imwrite('Sobel_edge_detection.png', sobelxy)

# 2 Canny edge detection
def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    # Canny Edge Detection
    edges = cv2.Canny(image=image, threshold1=threshold_1, threshold2=threshold_2)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    cv2.imwrite('Canny_edge_detection.png', edges)

    return edges

# 3 Template matching
def template_match(image, template):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('res.png', image_rgb)

# 4 Resize
def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == "up":
        # Upscale
        resized = cv2.pyrUp(image, dstsize=(image.shape[1] * scale_factor, image.shape[0] * scale_factor))
    elif up_or_down == "down":
        # Downscale
        resized = cv2.pyrDown(image, dstsize=(image.shape[1] // scale_factor, image.shape[0] // scale_factor))
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    cv2.imwrite(f"resized_{up_or_down}_{scale_factor}.png", resized)
    return resized



def main():
    # Read the original image
    img = cv2.imread('lambo.png')
    imgshape = cv2.imread('shapes-1.png', 0)
    template = cv2.imread('shapes_template.jpg', 0)
    cv2.waitKey(0)
    if img is None:
        print("Error: Image not found!")
        return

    # Display original image
    cv2.imshow('Original', img)

    # Grayscale and image blur
    processed = preprocessed_image(img)

    # Sobel edge detection
    sobel_edge_detection(processed)

    # Canny edge detection
    canny_edge_detection(processed)

    # Template matching
    template_match(imgshape, template)

    # Zoom in (scale factor 2)
    resize(img, scale_factor=2, up_or_down="up")

    # Zoom out (scale factor 2)
    resize(img, scale_factor=2, up_or_down="down")

if __name__ == '__main__':
    main()