import cv2

# Read image
image = cv2.imread("input.jpg")

if image is None:
    print("Image not found")
    exit()

# Resize
image = cv2.resize(image, (800, 600))

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blur, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Objects detected:", len(contours))

# Draw contours
output = image.copy()

cv2.drawContours(
    output,
    contours,
    -1,
    (0, 255, 0),
    2
)

# Display
cv2.imshow("Detected Objects", output)
cv2.imshow("Edges", edges)

cv2.imwrite("contours.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()