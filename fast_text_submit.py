import string
import re
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from fasttext import supervised, load_model

####################################################################
####################################################################

redundant_signs = set(string.punctuation) - set(['.'])
stop_words = list(set(stopwords.words('english')))
letters = [x for x in string.ascii_lowercase + '. ']
stemmer = PorterStemmer()

####################################################################
####################################################################


def clean_data(inp_str):
    inp_str = inp_str.lower()

    # fix haven't|doesn't|shouldn't cases
    inp_str = inp_str.replace("n't", " not")
    inp_str = inp_str.replace("'re'", " are")

    # here may be actor's names, types of smth etc. I guess it's redundant info
    # let's discuss of necessity of this block
    bracket_words = re.findall('([\(\[\{].+?[\)\]\}])', inp_str)
    for word in bracket_words:
        inp_str = inp_str.replace(''.join(word), "")

    # replace redundant_signs
    for item in redundant_signs:
        inp_str = inp_str.replace(item, ' ')

    # replace digits
    inp_str = re.sub('\d', ' ', inp_str)
    # replace two or more dots. 1 dot is remained as it separates sentences
    inp_str = re.sub('\.{1,10}', ' ', inp_str)
    # replace one-letter words or just letters
    inp_str = re.sub(r"\b[a-z]{1}\b", ' ', inp_str)

    return ' '.join(list(filter(None, inp_str.split(' '))))


def finalize_data(df):
    df['clean_text'] = df['text'].apply(clean_data)
    df['clean_text'] = df['clean_text'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
    df['is_ascii'] = df['clean_text'].apply(lambda x: set(x).issubset(letters))
    df['letters'] = df['clean_text'].apply(len)
    df['words'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['stemed_text'] = df['clean_text'].apply(lambda x: " ".join([stemmer.stem(w) for w in x.split()]))
    df['new_label'] = df['label'].apply(lambda x: '__label__1 ' if x == 1 else '__label__0 ')

    df = df[df['is_ascii'] == 1]
    df = df[df['letters'] > 0]
    df = df.reset_index()
    df = df.ix[:, ['new_label', 'stemed_text']]

    return df


def main():
    # Load your data and process it
    data = pd.read_csv('data_rt.csv', sep='|')  # here type your file name
    print('source data: ', data.shape)
    data = finalize_data(data)
    data['label'] = data['new_label'].apply(lambda x: int(x.strip()[-1]))
    print('cleaned data: ', data.shape)

    # load model and make predictions
    classifier = load_model('model_ft.bin', label_prefix='__label__')
    prediction = classifier.predict_proba(list(data['stemed_text']))
    train_predictions = [int(item[0][0]) for item in prediction]
    train_probabilities = [item[0][1] for item in prediction]

    print("=" * 30)
    print('****Results****')
    acc = accuracy_score(data['label'], train_predictions)
    print("Accuracy: {:.4%} \n".format(acc))


if __name__ == '__main__':
    main()
