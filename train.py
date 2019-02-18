#J Doran 2019
import json
import nltk
import nltk.lm

nltk.download('stopwords')
nltk.download('punkt')

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import TweetTokenizer
from nltk.lm import MLE

fp = open(input("Location of message.json: "))
json = json.load(fp)
messages = json["messages"]

ps = [p["name"] for p in json["participants"]]

name = input("Which participant (" + " or ".join(ps) + "): ")

n = int(input("n-gram size (probably 2 or 3): "))

maxLength = 20 #int(input("Maximum sentence length: "))

tkn = TweetTokenizer()
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
corpus = []
for m in messages:
    if m["sender_name"] == name:
        corpus += [tkn.tokenize(m["content"])]

train, vocab = padded_everygram_pipeline(n,corpus)
lm = MLE(n)
lm.fit(train, vocabulary_text=vocab)

def clean(wordList):
    out = ""
    for word in wordList:
        if word != '</s>' and word != '<s>':
            out += " "
            out += word
        else:
            if word == '</s>':
                break
    return out

print("Finished training! ('exit' to quit)")

previous = ""
while (previous != "exit"):
    previous = input(clean(lm.generate(maxLength, text_seed=['<s>'])))