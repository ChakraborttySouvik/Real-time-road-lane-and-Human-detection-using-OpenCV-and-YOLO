#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def grayscale(img):
    """Applies the Grayscale transform."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform."""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """Applies an image mask."""
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw lines on an image."""
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Returns an image with Hough lines drawn."""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """Combine two images."""
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_frame(image):
    gray_image = grayscale(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    gauss_gray = gaussian_blur(mask_yw_image, 5)
    canny_edges = canny(gauss_gray, 50, 150)
    
    imshape = image.shape
    vertices = np.array([[
        (imshape[1] / 9, imshape[0]),
        (imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10),
        (imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10),
        (imshape[1] - imshape[1] / 9, imshape[0])
    ]], dtype=np.int32)
    
    roi_image = region_of_interest(canny_edges, vertices)
    line_image = hough_lines(roi_image, 2, np.pi/180, 20, 50, 200)
    result = weighted_img(line_image, image)
    return result

# Process images
image_folder_path = r"C:\Users\souvi\OneDrive\Desktop\New folder (2)\test_images"
for source_img in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, source_img)
    image = mpimg.imread(image_path)
    processed = process_frame(image)
    output_path = os.path.join(image_folder_path, "annotated_" + source_img)
    mpimg.imsave(output_path, processed)
