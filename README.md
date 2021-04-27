# Denormalizer

This repository provides text denormalization models for English and Russian,
as described in the paper
_Benjamin Suter, Josef Novak: Neural Text Denormalization for Speech Transcripts (2021)_
(submitted at Interspeech 2021). The models are published under a BSD 3 licence.

Text denormalization includes prediction of punctuation, capitalization,
and transformation of number words into digits.

We provide small (`s`) and large (`l`) models for English (`en`) and Russian (`ru`).
The large models have consistently better performance metrics, but the small models
provide doubled inference speed at a reasonable quality (see paper for details).

The character range for the input strings is defined as the following sets:
- English: `abcdefghijklmnopqrstuvwxyz' `
- Russian: `abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя `

Other characters are possible in the input, but they may distort the output unpredictably.


## 1. Setup

Please install [fairseq](https://github.com/pytorch/fairseq) from source:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Additionally, you need to install subword-nmt and moses:
`pip install subword_nmt sacremoses`


## 2. Usage

The denormalizer models can be accessed with the script `denormalize.py`.

The script takes either a string or a file:
* `--string`: String to denormalize.
* `--file`: File to denormalize.

In the absence of both arguments, the script starts in the interactive mode.

It takes the following named arguments:
* `--lang`: Model language (`en` or `ru`). Defaults to `en`.
* `--size`: Model size (`s` or `l`). Defaults to `l`.
* `--outfile`: File to which the output will be written. Defaults to `stdout`.
* `--beam`: Beam size. Defaults to `5`.


### 2.1 Example Usage

For the interactive mode, use:
* `python denormalize.py --lang <LANG> --size <SIZE>`

In order to denormalize a full file, use:
* `python denormalize.py --lang <LANG> --size <SIZE> --file <FILE> --outfile <OUTFILE>`

In order to denormalize a single string, use:
* `python denormalize.py --lang <LANG> --size <SIZE> --string <STRING>`
