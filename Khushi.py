import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the color grading map image
image = cv2.imread('color_grading_map.jpg')  # Replace 'color_grading_map.jpg' with your image path

# Convert the image from BGR to RGB (OpenCV loads images in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a list of pixels
pixels = image_rgb.reshape(-1, 3)

# Define the number of colors/clusters you want to identify
num_clusters = 5  # You can adjust this based on your requirements

# Apply K-Means clustering to identify the dominant colors
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get the RGB values of the cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_

# Convert the RGB values to integers
dominant_colors = dominant_colors.astype(int)

# Print the dominant colors
for color in dominant_colors:
    print(f"RGB: {color[0]}, {color[1]}, {color[2]}")

# Optional: Display the dominant colors as swatches
swatches = np.zeros((100, num_clusters, 3), dtype=np.uint8)
for i, color in enumerate(dominant_colors):
    swatches[:, i, :] = color

cv2.imshow('Dominant Colors', swatches)
cv2.waitKey(0)
cv2.destroyAllWindows()
