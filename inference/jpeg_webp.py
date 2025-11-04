from PIL import Image
import os

input_folder = "./jpeg_webp"
output_folder = "./inference_image"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    #  jpg / jpeg / webp
    if file.lower().endswith((".jpg", ".jpeg", ".webp")):
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path).convert("RGB")
        new_name = os.path.splitext(file)[0] + ".png"
        img.save(os.path.join(output_folder, new_name), "PNG")

print("Finish!")
