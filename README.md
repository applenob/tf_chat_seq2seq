# Seq2Seq Chat-bot Baseline

A Seq2Seq chat-bot baseline implemented by Tensorflow.

- Python: 3.6
- Tensorflow: 1.12

## Usage:

First clone this repository to your computer and install dependencies:

```
git clone https://github.com/applenob/tf_chat_seq2seq.git
cd tf_chat_seq2seq
pip install -r requirements.txt
```

### Prepare the Dataset

Download dialogue data from [Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus).

```
git clone https://github.com/candlewill/Dialog_Corpus.git input
```

Then `unzip` the data.

Make sure we can visit data by path like `input/xiaohuangji50w_nofenci.conv`.

### Train the Model

First you can change the configure in file `config.yml`.

Generate vectorized data:

```
python data_process.py --conv_path input/xiaohuangji50w_nofenci.conv
``` 

Train Tensorflow version seq2seq model:

```
python main.py --mode train
```

Have fun with the model you trained:

```
python main.py --mode predict
```