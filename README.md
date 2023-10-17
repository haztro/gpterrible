# gpterrible

Terrible GPT inference with minimal abstraction for my own learning. Uses the model in Karpathy's [nanoGPT lecture](https://github.com/karpathy/ng-video-lecture).

Train the model found in utils/gpt.py on Tiny Shakespeare to generate 'model.pt':
```bash
python gpt.py
```

Convert 'model.pt' to 'model.json' with utils/pt_to_json.py lmao:
```bash
python pt_to_json.py
```

Create tokenizer with utils/tokenizer.py:
```bash
python tokenizer.py
```

Generate tiny shakespeare (slowly):
```bash
python gpterrible.py
```
