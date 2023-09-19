import nltk
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'

nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

moby_dick = gutenberg.raw('melville-moby_dick.txt')
tokens = word_tokenize(moby_dick)
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
pos_tags = pos_tag(filtered_tokens)
pos_freq = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for (word, tag) in pos_tags[:20]]

pos_freq.plot(30, cumulative=False)

print("Top 5 Parts of Speech:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

print("\nLemmatized Tokens:")
print(lemmatized_tokens)

sia = SentimentIntensityAnalyzer()

sentiments = [sia.polarity_scores(sentence)['compound'] for sentence in nltk.sent_tokenize(moby_dick)]
average_sentiment = sum(sentiments) / len(sentiments)
print("\nAverage Sentiment Score:", average_sentiment)

if average_sentiment > 0.05:
    print("The overall text sentiment is positive.")
else:
    print("The overall text sentiment is negative.")

plt.show()
