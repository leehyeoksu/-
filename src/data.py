from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader

class setdata(Dataset):
    def __init__(self,data,tr=None):
        super().__init__()
        self.data=data
        self.tf=tr
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x=self.data['image'][index].convert("RGB")
        y=self.data['label'][index]
        if self.tf:
            x=self.tf(x)
        x = x.clone()
        return x,y
        