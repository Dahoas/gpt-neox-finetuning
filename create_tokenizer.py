from tokenizers import Tokenizer
from megatron.tokenizer.tokenizer import HFTokenizer


filepath = 'data/tokenizer/20B_tokenizer.json'
tokenizer = HFTokenizer(filepath)
hello_ids = tokenizer.tokenize("hello")
print(hello_ids)

ids = tokenizer.tokenize('}{')
print(ids)
ids = tokenizer.tokenize('---')
print(ids)
start_ids = tokenizer.tokenize('<|startoftext|>')
end_ids = tokenizer.tokenize('<|endoftext|>')
print(start_ids)
print(end_ids)