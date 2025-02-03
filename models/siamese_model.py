import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
    Реализует Siamese сеть для обучения эмбеддингов изображений.

    Эта сеть состоит из двух идентичных ветвей, предназначенных для извлечения признаков из двух входных изображений.
    Затем эти признаки используются для вычисления loss функции, которая обучает сеть создавать схожие эмбеддинги для похожих
    изображений и разные эмбеддинги для непохожих.

    Атрибуты:
        embedding_dim (int): Размерность эмбеддинга (векторного представления) изображения. По умолчанию 128.
        conv_layers (nn.Sequential): Последовательность сверточных слоев для извлечения признаков.
        fc_layers (nn.Sequential):  Последовательность полносвязных слоев для получения эмбеддинга заданной размерности.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.conv_layers = nn.Sequential(
            # (batch_size, 3, 128, 128)
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # (batch_size, 64, 128, 128)
            nn.ReLU(), # (batch_size, 64, 128, 128)
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 64, 64, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (batch_size, 128, 64, 64)
            nn.ReLU(), # (batch_size, 128, 64, 64)
            nn.MaxPool2d(kernel_size=2, stride=2), # (batch_size, 128, 32, 32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (batch_size, 256, 32, 32)
            nn.ReLU(), # (batch_size, 256, 32, 32)
            nn.MaxPool2d(kernel_size=2, stride=2) # (batch_size, 256, 16, 16)
        )
        
        flattened_size = 256 * 16 * 16
        
        self.fc_layers = nn.Sequential(
            # (batch_size, flattened_size)
            nn.Linear(flattened_size, 512), # (batch_size, 512)
            nn.ReLU(), # (batch_size, 512)
            nn.Linear(512, embedding_dim) # (batch_size, embedding_dim)
        )

    def forward_once(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc_layers(x)
        return x
        
    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2