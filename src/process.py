import json
import os
import numpy as np
from PIL import Image

def process_split(annotations_path, product_images_dir, logo_images_dir, output_prefix, resize_size=(128, 128)):

    """
    Обрабатывает аннотации, обрезает изображения продуктов на основе ограничивающих рамок, 
    изменяет размеры как обрезанных изображений продуктов, так и изображений логотипов, 
    и сохраняет их в виде массивов NumPy.

    Args:
        annotations_path (str): Путь к файлу JSON, содержащему аннотации.
        product_images_dir (str): Каталог, содержащий изображения продуктов.
        logo_images_dir (str): Каталог, содержащий изображения логотипов.
        output_prefix (str): Префикс для выходных файлов массивов NumPy.
        resize_size (tuple, optional): Размер (ширина, высота), до которого изменяются размеры изображений. По умолчанию (128, 128).

    Returns:
        tuple: Кортеж, содержащий два массива NumPy:
               - product_crops_np: Массив NumPy обрезанных и измененных по размеру изображений продуктов.
               - logo_arrays_np: Массив NumPy измененных по размеру изображений логотипов.
    """

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    product_crops = []
    logo_arrays = []
    
    for product_filename in annotations:
        product_path = os.path.join(product_images_dir, product_filename)
        if not os.path.exists(product_path):
            print(f"Warning: Product image {product_filename} not found. Skipping.")
            continue
        
        try:
            product_img = Image.open(product_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {product_filename}: {e}")
            continue
        
        for bbox, logo_filename in annotations[product_filename]:
            if logo_filename == "__unknown__":
                continue 
            
            logo_path = os.path.join(logo_images_dir, logo_filename)
            if not os.path.exists(logo_path):
                print(f"Warning: Logo image {logo_filename} not found. Skipping.")
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            img_width, img_height = product_img.size
            
            if (x1 >= x2 or y1 >= y2 or
                x1 < 0 or y1 < 0 or
                x2 > img_width or y2 > img_height):
                print(f"Invalid bbox {bbox} in {product_filename}. Skipping.")
                continue
            
            try:
                crop = product_img.crop((x1, y1, x2, y2))
            except Exception as e:
                print(f"Error cropping {product_filename}: {e}")
                continue
            
            crop_resized = crop.resize(resize_size, Image.Resampling.LANCZOS)
            crop_array = np.array(crop_resized, dtype=np.uint8)
            
            try:
                logo_img = Image.open(logo_path).convert('RGB')
            except Exception as e:
                print(f"Error opening logo {logo_filename}: {e}")
                continue
            
            logo_resized = logo_img.resize(resize_size, Image.Resampling.LANCZOS)
            logo_array = np.array(logo_resized, dtype=np.uint8)
            
            product_crops.append(crop_array)
            logo_arrays.append(logo_array)
    
    product_crops_np = np.array(product_crops)
    logo_arrays_np = np.array(logo_arrays)
    
    assert len(product_crops_np) == len(logo_arrays_np), "Mismatch in number of product crops and logo images."
    
    np.save(f"{output_prefix}_product_crops.npy", product_crops_np)
    np.save(f"{output_prefix}_logo_images.npy", logo_arrays_np)
    print(f"Saved {len(product_crops_np)} pairs for {output_prefix} split.")

    return product_crops_np, logo_arrays_np