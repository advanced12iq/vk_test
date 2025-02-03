import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LogoDataset(Dataset):
    """
    Atributes:
        product_crops_path (str): Путь к файлу NumPy (.npy), содержащему обрезанные изображения продуктов.
        logo_images_path (str): Путь к файлу NumPy (.npy), содержащему изображения логотипов.
        transform (callable, optional): Функция преобразования (например, из torchvision.transforms),
                                      которая будет применена к изображениям. По умолчанию None.
        product_crops (numpy.ndarray): Массив NumPy с обрезанными изображениями продуктов, загруженный из файла.
        logo_images (numpy.ndarray): Массив NumPy с изображениями логотипов, загруженный из файла.
    """
    def __init__(self, product_crops_path, logo_images_path, transform=None):

        self.product_crops = np.load(product_crops_path)
        self.logo_images = np.load(logo_images_path)
        self.transform = transform

        assert len(self.product_crops) == len(self.logo_images), "Несоответствие в количестве обрезанных изображений продуктов и изображений логотипов."

    def __len__(self):
        return len(self.product_crops)
        
    def __getitem__(self, index):

        crop_image = self.product_crops[index]
        logo_image = self.logo_images[index]

        crop_image = Image.fromarray(crop_image).convert("RGB")
        logo_image = Image.fromarray(logo_image).convert("RGB")

        if self.transform:
            crop_image = self.transform(crop_image)
            logo_image = self.transform(logo_image)
        
        return crop_image, logo_image