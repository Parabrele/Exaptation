import datasets
import torch
from easy_transformer import EasyTransformer
from easy_transformer.utils import tokenize_and_concatenate

if __name__ == "__main__":
    print("Loading model...")

    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

    print("Loading dataset...")

    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")

    print("Tokenizing dataset...")

    print(dataset[0]['text'][:100])
    tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=256, column_name="text", add_bos_token=True, num_proc=1)

    print("Creating dataloader...")

    data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=8, shuffle=True, pin_memory=True)

    print("Starting training...")

    for batch in data_loader :
        print(batch.keys())
        break
