import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from datasets import load_dataset
import re
import sys

class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: List[str]) -> List[List[float]]:

        # big batches are causing us to run out of memory. Limit the size
        embeddings = list()
        for i in range(0, len(text), 100):
            end = i + 100
            if end > len(text):
                end = len(text)
            batch = text[i:end]
        
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
            embeddings.extend(self.model(
                    **inputs,
                    decoder_input_ids=decoder_inputs['input_ids'],
                    encode_only=True,
                ).to('cpu').tolist())
        
        return embeddings

    @torch.no_grad()
    def generate_from_latent(self, latent: List[float], max_length=512, temperature=1.0) -> str:
        dummy_text = ['.']
        dummy = torch.tensor(self.embed(dummy_text)).to(device)
        latent = torch.tensor(latent).to(device)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def get_sentences(article):
    # split the article into sentences
    sentences = re.split(r'(?<=[.!?;:\n\r])\s+', article)
    # remove empty sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    # remove sentences that are too long by filtering out sentences with more than 400 words
    sentences = [s for s in sentences if len(s.split()) <= 400]
    return sentences

# remove any rows that only have a singe sentence or not sentences.
def has_multiple_sentences(item):
    return len(item['sentences']) > 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
autoencoder = BottleneckT5Autoencoder(model_path='thesephist/contra-bottleneck-t5-large-wikipedia', device=device)

def get_embeddings_batch(sentences):
    return autoencoder.embed(sentences)

#
# Usage: python process_wiki_text.py [dataset name] [split] [output name]
#
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print('Usage: python process_wiki_text.py [dataset name] [split] [output name]')
        exit(-1)

    dataset_name = sys.argv[1]
    split_name = sys.argv[2]
    save_name = sys.argv[3]

    wiki = load_dataset(path="wikitext", name=dataset_name, split=split_name)

    wiki_sentences = wiki.map(lambda x: {'sentences': get_sentences(x['text'])})
    print(len(wiki_sentences))
    wiki_sentences = wiki_sentences.filter(has_multiple_sentences)
    print(len(wiki_sentences))

    wiki_text_embeddings = wiki_sentences.map(lambda x: {'embeddings': get_embeddings_batch(x['sentences'])})
    wiki_text_embeddings.save_to_disk(save_name)

