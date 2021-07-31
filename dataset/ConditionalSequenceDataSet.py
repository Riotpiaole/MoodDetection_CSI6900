from torch.utils.data import Dataset , DataLoader, dataloader
from torch import tensor
from transformers import BertTokenizer
import pickle
from pdb import  set_trace
from time import time 
from tqdm import tqdm


class ConditionalSequenceDataset(Dataset):
    def __init__(self, dataset_input_id , lm_mask, dataset_emo_label ) -> None:
        super().__init__()
        # with open(f"")
        self.input_ids = dataset_input_id
        self.lm_mask = lm_mask
        self.emo_label = dataset_emo_label
        

        # for (lm_ids, mask,  emo ) in zip( masked_lm_text, masked_label, emo_tensor):
        #     self.dataset.append( (lm_ids, mask , emo))
    
    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, index: int) :
        masked_lm_text = self.input_ids[index]
        masked_label, attention_mask = self.lm_mask[index]
        emo_token = self.emo_label[index]
        return tensor(masked_lm_text),tensor(attention_mask), tensor(masked_label) , tensor(emo_token)

if __name__ == "__main__":
    from constants import LMBERT_PKL 
    with open(LMBERT_PKL, 'rb') as handler:
        dicts = pickle.load(handler)
    
    # with open(EMBEDDING_PKL, 'rb') as handler:
    #     bert_dict = pickle.load(handler)
    train_data, test_data , val_data = dicts['train_dataset'], dicts['test_dataset'] , dicts['val_dataset']
    
    dataset = ConditionalSequenceDataset(*train_data)
    print("Creating dataloader.... ", end=' ')
    loader = DataLoader(dataset, 32, False)
    print("done")

