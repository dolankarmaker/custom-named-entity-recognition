import torch
from transformers import BertTokenizer, BertConfig
from sklearn.model_selection import train_test_split

data_root = "data_directory/"

MAX_LEN = 75
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.device_count())
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


