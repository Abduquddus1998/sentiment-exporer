import nltk
from rake_nltk import Rake

nltk.download('stopwords')
nltk.download('punkt')


def extract_key_phrases(text):
    rake_nltk_var = Rake()

    rake_nltk_var.extract_keywords_from_text(text)

    ranked_phrases_with_scores = rake_nltk_var.get_ranked_phrases_with_scores()

    key_phrases = [{"score": score, "phrase": phrase} for score, phrase in ranked_phrases_with_scores]

    return key_phrases
