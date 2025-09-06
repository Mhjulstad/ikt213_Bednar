import numpy as np
import cv2


def print_image_information(image):
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    print("Image dimensions:")
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)


def main():
    img = cv2.imread('lena-1.png')  # load in color
    if img is None:
        print("Error: Image not found!")
        return
    print_image_information(img)

if __name__ == '__main__':
    main()