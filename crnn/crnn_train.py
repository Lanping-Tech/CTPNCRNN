import torchvision.transforms as transforms
import torch
import torchvision
import numpy as np

from Chinese_alphabet import alphabet
import crnn_utils as utils
from crnn_model import CRNN
from crnn_config import *

class CRNN_Trainer:
    def __init__(self):

        self.nclass = len(alphabet) + 1
        self.nc = 1
        self.converter = utils.strLabelConverter(alphabet, ignore_case=True)
        self.criterion = torch.nn.CTCLoss()

        self.crnn = CRNN(imgH, nc, self.nclass, nh)
        self.optimizer = torch.optim.Adam(self.crnn.parameters(), lr=lr, betas=(beta1, 0.999))
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(imgH, imgW)),
            transforms.ToTensor()]) 



    def train_batch(self, img, boxes, texts):
        
        self.crnn.train()
        sub_imgs = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            sub_img = img[y1:y2, x1:x2]
            sub_img = sub_img.astype(np.uint8)
            sub_img = self.data_transforms(sub_img)
            sub_imgs.append(sub_img)

        batch_size = len(sub_imgs)
        sub_imgs = torch.stack(sub_imgs)


        text = torch.IntTensor(batch_size * 10)  # 假设每个句子长为5
        length = torch.IntTensor(batch_size)
        t, l = self.converter.encode(texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = self.crnn(sub_imgs)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = self.criterion(preds, text, preds_size, length)
        self.crnn.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def save_model(self, path):
        torch.save(self.crnn.state_dict(), path)


if __name__ == '__main__':
    img = np.random.rand(32, 32, 3) * 255
    img = img.astype(np.uint8)
    boxes = np.array([[1, 10, 20, 20], [1, 10, 20, 20]])
    texts = ['a', 'b']
    trainer = CRNN_Trainer()

    loss = trainer.train_batch(img, boxes, texts)
    print(loss)


