import os
import random
from PIL import Image


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        raise ValueError(f"Image too small for crop: {w}x{h} < {size}x{size}")
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def build_grid(images, grid_size: int, tile_size: int) -> Image.Image:
    grid_img = Image.new("RGB", (grid_size * tile_size, grid_size * tile_size))
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        grid_img.paste(img, (col * tile_size, row * tile_size))
    return grid_img


def main():
    input_dir = r"D:\AI_Code\DeepLearing\PersonalProject\Pycharm_Workplace\PycharmProjects\SAR_VehicleTargetSampleAugmentationBasedOnGan\MSTAR\PERSONAL_MSTAR\15_DEG"
    output_dir = r"D:\AI_Code\DeepLearing\PersonalProject\Pycharm_Workplace\PycharmProjects\SAR_VehicleTargetSampleAugmentationBasedOnGan\Test_Results\Test_RealImage_Results"
    os.makedirs(output_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(root, f))

    grid_size = 8
    tile_size = 128
    needed = grid_size * grid_size
    if len(image_paths) < needed:
        raise RuntimeError(f"Not enough images: {len(image_paths)} < {needed}")

    num_grids = 5
    for i in range(num_grids):
        selected = random.sample(image_paths, needed)
        tiles = []
        for p in selected:
            with Image.open(p) as img:
                img = img.convert("RGB")
                tiles.append(center_crop(img, tile_size))

        grid_img = build_grid(tiles, grid_size, tile_size)
        out_path = os.path.join(output_dir, f"mstar_15deg_grid_8x8_{i+1:02d}.png")
        grid_img.save(out_path)


if __name__ == "__main__":
    main()
