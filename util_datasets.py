from torch.utils.data import Dataset, DataLoader
###### Class dataset
class MyDataset(Dataset):
    def __init__(self, x,y):
        super(Dataset,self).__init__()
        self.x = x
        if y is not None:
            self.y = y
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        else:
            return self.x[index]