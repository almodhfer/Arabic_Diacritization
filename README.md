## Implementation of several deep learning models for recovering Arabic language diacritics

This is the Pytorch implementation of the models as described in my paper 
[Effective Deep Learning Models for Automatic Diacritization of Arabic Text](https://ieeexplore.ieee.org/document/9274427).
This work was done as part of my thesis 
(Developing a High-Quality Tool for Arabic Text-To-Speech Using Deep Learning Techniques) at Computer Science department, Qassim University.

There are four models as follows:

- The baseline model (baselin): consists of 3 bidirectional LSTM layers with optional batch norm layers.
- The seq2seq with attention model (seq2seq): uses the baseline model as an encoder and the location-sensitive attention.
- The Tacotron based seq2seq model (tacotron_based): uses the same encoder as Tacotron,
  but the decoder and the attention are the same as the seq2seq model.
- The CBHG model (cbhg): uses only the encoder of the Tacotron based model with optional post LSTM, and batch norm layers.

The CBHG model achieves the best WER and DER with and without
case_ending.

# Data Preprocessing

- The data can either be processed before training or when loading each batch.
- If you decide to process the corpus before training, then the corpus must have test.csv, train.csv, and valid.csv. Each file must contain three columns: text (the original text), text without diacritics, and diacritics. You have to define the column separator and diacritics separator in the config file.
- If the data is not preprocessed, you can specify that in the config.
  In that case,  each batch will be processed and the text and diacritics 
  will be extracted from the original text.
- You also have to specify the text encoder and the cleaner functions.
  This work includes two text encoders: BasicArabicEncoder, ArabicEncoderWithStartSymbol.
  Moreover, we have one cleaning function: valid_arabic_cleaners, which clean all characters except valid Arabic characters,
  which include Arabic letters, punctuations, and diacritics.

# Training

All models config are placed in the config directory.

```bash
python train.py --model model_name --config config/config_name.yml
```

The model will report the WER and DER while training using the
diacritization_evaluation package. The frequency of calculating WER and
DER can be specified in the config file.

# Testing

The testing is done in the same way as training:

```bash
python test.py --model model_name --config config/config_name.yml
```

The model will load the last saved model unless you specified it in the config:
test_data_path. If the test file name is different than test.csv, you
can add it to the config: test_file_name.

### Issues 
If you find any problem with the code or have any suggestion please submit an issue and I will resolve it as soon as possible.

### Citation

Please cite our paper if you use this repository:

```text
M. A. H. Madhfar and A. M. Qamar, "Effective Deep Learning Models for Automatic Diacritization of Arabic Text," in IEEE Access, vol. 9, pp. 273-288, 2021, doi: 10.1109/ACCESS.2020.3041676.

```
