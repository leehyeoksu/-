from torchvision.transforms import transforms
from torchvision import transforms
from torch.utils.data import Dataset

class setdata(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data['image'][index].convert("RGB")
        y = self.data['label'][index]

        x = self.tf(x)
        return x, y