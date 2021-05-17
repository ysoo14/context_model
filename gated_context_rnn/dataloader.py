import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd


class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index): # 0: utterance textual feature, 1: sentence length, 2: label, 3: speaker information 4: m_list 5: f_list 6: vid
        vid = self.keys[index]
        m, f = self.getSpeakerList(vid)
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               torch.LongTensor([0 if x == 'M' else 1 for x in self.videoSpeakers[vid]]),\
               vid

    def getSpeakerList(self, vid):
        speaker_list = torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.videoSpeakers[vid]])
        male_list =[]
        female_list = []
  
        idx = 0
        for speaker in speaker_list:
            s = torch.argmax(speaker)
            s = s.cpu().detach().numpy()

            if s == 0:
                male_list.append(idx)
            else:
                female_list.append(idx)

            idx+=1
        return male_list, female_list

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<4 else dat[i].tolist() for i in dat]

if __name__ == '__main__':
    dataset = IEMOCAPDataset()
