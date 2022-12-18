import numpy as np

class Dataloader:
    def __init__(self):
        self.embeddings = np.load('hara_data/embeddings.npy')
        self.img_paths = list(np.load('hara_data/img_paths.npy'))
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, i):
        return self.embeddings[i], self.img_paths[i]
    
    def load_img(self, path):
        if path not in self.img_paths:
            raise ValueError(f'{path} is not in self.img_paths')
        return np.load(path)

if __name__ == '__main__':
    dataloader = Dataloader()
    for i in dataloader:
        print(i)