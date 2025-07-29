from PIL import Image

img = Image.open('imagetobyteImage/vs.png')

# Ensure it's in RGB mode
img = img.convert("RGB")

# Get pixel data
data = list(img.getdata())

# Invert pixel values
inverted_data = [(255 - r, 255 - g, 255 - b) for (r, g, b) in data]

# Convert inverted data to bytes
byte_data = bytes([value for pixel in inverted_data for value in pixel])

print(byte_data)  # or write to a file