import PIL.Image as Image
import torch.utils.data as data
from glob import glob

NO_DR_PATH = "./data/No DR/"
MILD_PATH = "./data/Mild/"
MODERATE_PATH = "./data/Moderate/"
SEVERE_PATH = "./data/Severe/"
PDR_PATH = "./data/Proliferative DR/"
ENCODE = {"nodr": 0, "mild": 1, "moderate": 2, "severe": 3, "pdr": 4}


def loadfile_name(pth, ext='.jpeg'):
    return glob(pth + '*' + ext)

class MildModerateData(data.Dataset):
    # Raw Dataset
    def __init__(self, aug,
                 MILD_PATH=MILD_PATH,
                 MODERATE_PATH=MODERATE_PATH,
                 SEVERE_PATH=SEVERE_PATH,
                 NO_DR_PATH=NO_DR_PATH,
                 PDR_PATH=PDR_PATH,
                 balance=True,
                 ):
        # super(covdata, self).__init__()
        self.MODERATE_PATH = MODERATE_PATH
        self.MILD_PATH = MILD_PATH
        self.SEVERE_PATH=SEVERE_PATH
        self.NO_DR_PATH=NO_DR_PATH
        self.PDR_PATH=PDR_PATH
        self.labels = {0: 'nodr', 1: 'mild', 2: 'moderate', 3: 'severe', 4: 'pdr'}
        self.items = list()
        self.aug = aug

        Milds = loadfile_name(MILD_PATH)
        Moderats = loadfile_name(MODERATE_PATH)
        Severes = loadfile_name(SEVERE_PATH)
        Nodrs = loadfile_name(NO_DR_PATH)
        Pdrs = loadfile_name(PDR_PATH)
        m = min([len(x) for x in [Milds, Moderats, Severes, Nodrs,Pdrs]])
        for i, ds in enumerate([Milds, Moderats,Severes,Nodrs ,Pdrs]):
            ds = ds[:m] if balance else ds
            self.items += [{'im_path': item,
                            'label': self.labels[i]} for item in ds]

    def __len__(self):
        return self.items.__len__()

    def __getitem__(self, i):
        obj =  self.items[i]
        im = Image.open(obj['im_path'])
        lb = ENCODE[obj['label']]
        return self.aug(im), lb

def create_loader(dts, bs, ):
    tdlr = data.DataLoader(dts, bs, shuffle=True,)
    return tdlr


def getloaders(aug, trtes=[.7, .1, .2], bs=128):
    dts = MildModerateData(aug)
    tl = dts.__len__()
    slen = [int(x * tl) for x in trtes[:2]]
    slen.append(tl - sum(slen))
    tr, va, te = data.random_split(dts, slen)
    return [create_loader(x, bs) for x in [tr, va, te]]
