import os
import torch
from PIL import Image
from src import (
    config
)
from tqdm import tqdm

def predict_logo(model, product_path, logos_dir, threshold=0.7, device=config.DEVICE):
    """
    Предсказывает, содержит ли изображение продукта логотип из заданного набора логотипов.

    Аргументы:
        model (torch.nn.Module): Обученная модель Siamese сети.
        product_path (str): Путь к изображению продукта.
        logos_dir (str): Каталог, содержащий изображения логотипов.
        threshold (float, optional): Порог схожести для определения наличия логотипа. По умолчанию 0.7.
        device (str, optional): Устройство для выполнения инференса. По умолчанию config.DEVICE.

    Возвращает:
        dict: Словарь, содержащий максимальную схожесть, порог, статус обнаружения и схожесть для каждого логотипа.
    """

    try:
        product_img = Image.open(product_path).convert('RGB')
        product_img = config.TRANSFORM_PRODUCT(product_img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading product image: {e}")
        return {"max_similarity": 0.0, "threshold": threshold, "detected": False, "similarities":{}}


    similarities = {}
    max_similarity = 0
    detected = False

    logo_files = [f for f in os.listdir(logos_dir) if os.path.isfile(os.path.join(logos_dir, f))]
    for logo_file in tqdm(logo_files, desc="Checking logos"):
        try:
            logo_path = os.path.join(logos_dir, logo_file)
            logo_img = Image.open(logo_path).convert('RGB')
            logo_img = config.TRANSFORM_LOGO(logo_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output1, output2 = model(product_img, logo_img)
                similarity = torch.sigmoid(torch.sum(torch.abs(output1 - output2))).item()
                similarities[logo_file] = similarity
                max_similarity = max(max_similarity, similarity)
                if similarity > threshold:
                    detected = True
        except Exception as e:
            print(f"Error loading logo image: {logo_file}, {e}")
            continue


    return {"max_similarity": max_similarity, "threshold": threshold, "detected": detected, "similarities":similarities}