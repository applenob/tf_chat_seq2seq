# Seq2Seq Chat-bot Baseline

A Seq2Seq chat-bot baseline implemented by Tensorflow.

- Python: 3.6
- Tensorflow: 1.12

## Usage:

Generate vectorized data:

```
python data_process.py
``` 

Train Tensorflow version seq2seq model:

```
python main.py --mode train
```

Have fun with the model you trained:

```
python main.py --mode predict
```