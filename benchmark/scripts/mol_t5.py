from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from torch.utils.data import DataLoader


def main():
    tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-smiles2caption", model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-smiles2caption')
    model.to('cuda')
    model.eval()

    # test_df = pd.read_csv('chefv2_pubmed_scaffold_test_with_pubmed_smiles.csv')
    test_df = pd.read_csv('opentargets_smiles_indications_20240626_pubchem_smiles.csv')

    # smiles_list = test_df['pubchem_smiles'].tolist()
    smiles_list = test_df['smiles'].tolist()

    # create dataloader

    class ForT5Dataset(torch.utils.data.Dataset):
        def __init__(self, tokenized):
            self.tokenized = tokenized
        
        def __len__(self):
            return len(self.tokenized["input_ids"])
        
        def __getitem__(self, index):
            input_ids = torch.tensor(self.tokenized["input_ids"][index]).squeeze()
            attention_mask = torch.tensor(self.tokenized["attention_mask"][index]).squeeze()
            
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenized = tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
    dataset = ForT5Dataset(tokenized)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    outputs = []
    for count, batch in enumerate(dataloader):
        print(f"Batch {count+1}/{len(dataloader)}, {torch.cuda.memory_allocated()} bytes used", end='\r')
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        with torch.no_grad():
            b_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, max_length=512)
            b_outputs = b_outputs.cpu().numpy()
        outputs.extend(tokenizer.batch_decode(b_outputs, skip_special_tokens=True))


    test_df['molt5_caption'] = outputs
    test_df.to_csv('molt5_opentargets_20240626.csv', index=False)


if __name__ == '__main__':
    main()