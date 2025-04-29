import re
import string
import nltk
from nltk.corpus import stopwords
import emoji
from collections import Counter
from autocorrect import Speller
import pandas as pd

nltk.download('stopwords')

class TextPreprocessor:
    """
    A class to preprocess and clean text data for various text mining and NLP tasks.
    Includes methods for handling contractions, emojis, special characters, multi-language support, and more.
    """

    def __init__(self, language="english", remove_stopwords=True, remove_emojis=True, handle_contractions=True):
        """
        Initialize the TextPreprocessor class.

        :param language: Language for stopword filtering (default is "english").
        :param remove_stopwords: Whether to remove stopwords (default is True).
        :param remove_emojis: Whether to remove emojis (default is True).
        :param handle_contractions: Whether to expand contractions (default is True).
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_emojis = remove_emojis
        self.handle_contractions = handle_contractions
        self.stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        self.speller = Speller()
        self.contractions = self._get_contractions()

    def _get_contractions(self):
        """
        Returns a dictionary of contractions for multiple languages.
        """
        return {
          "english": {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "couldn't": "could not", 
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
        "hasn't": "has not", "haven't": "have not", "isn't": "is not", "it's": "it is", 
        "let's": "let us", "mustn't": "must not", "needn't": "need not", "shouldn't": "should not", 
        "wasn't": "was not", "weren't": "were not", "won't": "will not", "wouldn't": "would not",
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is", 
        "we're": "we are", "they're": "they are", "i've": "i have", "you've": "you have", "he's": "he has", 
        "she's": "she has", "it's": "it has", "we've": "we have", "they've": "they have", "i'll": "i will", 
        "you'll": "you will", "he'll": "he will", "she'll": "she will", "we'll": "we will", 
        "they'll": "they will", "i'd": "i would", "you'd": "you would", "he'd": "he would", 
        "she'd": "she would", "we'd": "we would", "they'd": "they would", "i've": "i have", 
        "you've": "you have", "he's": "he has", "she's": "she has", "we've": "we have", 
        "they've": "they have", "i'd": "i would", "you'd": "you would", "he'd": "he would",
        "she'd": "she would", "we'd": "we would", "they'd": "they would"
    },
    "spanish": {
        "no es": "no está", "no está": "no está", "no quiero": "no quiero", "no puedo": "no puedo",
        "estoy": "estaba", "tengo": "tenía", "eres": "eras", "fue": "era", "seré": "sería", 
        "haré": "haría", "irá": "iría", "sería": "sería", "serás": "serías", "estará": "estaría",
        "hemos": "habíamos", "tenemos": "teníamos", "has": "habías", "puedo": "podría", "puedes": "podrías", 
        "hacer": "hacía", "debería": "debería", "soy": "fui", "estábamos": "estuvimos", "hicimos": "hacíamos", 
        "no he": "no estaba", "me he": "me estaba", "te he": "te estaba", "te": "te estaba", "esos": "aquellos",
        "estos": "estos estaban", "hay": "había", "tú": "te", "usted": "lo"
    },
    "german": {
        "ich bin": "ich bin", "ist nicht": "ist nicht", "ich habe": "ich habe", "du bist": "du bist",
        "wir sind": "wir sind", "sie sind": "sie sind", "ich kann": "ich könnte", "du kannst": "du könntest",
        "er kann": "er könnte", "sie kann": "sie könnte", "wir können": "wir könnten", "sie können": "sie könnten",
        "ich werde": "ich würde", "du wirst": "du würdest", "er wird": "er würde", "wir werden": "wir würden",
        "sie werden": "sie würden", "ich habe": "ich hatte", "du hast": "du hattest", "er hat": "er hatte",
        "wir haben": "wir hatten", "sie haben": "sie hatten", "ich würde": "ich würde", "du würdest": "du würdest",
        "wir würden": "wir würden", "sie würden": "sie würden", "ich habe": "ich hatte", "du hast": "du hattest",
        "wir haben": "wir hatten", "sie haben": "sie hatten", "es gibt": "es gab"
    },
    "french": {
        "je ne suis pas": "je suis", "je n'ai pas": "je ai", "je ne veux pas": "je veux", "tu n'as pas": "tu as",
        "tu ne veux pas": "tu veux", "il n'est pas": "il est", "elle n'est pas": "elle est", "nous ne sommes pas": "nous sommes",
        "vous n'êtes pas": "vous êtes", "ils ne sont pas": "ils sont", "elles ne sont pas": "elles sont", "je ne peux pas": "je peux",
        "tu ne peux pas": "tu peux", "nous ne pouvons pas": "nous pouvons", "vous ne pouvez pas": "vous pouvez", "ils ne peuvent pas": "ils peuvent",
        "je ne sais pas": "je sais", "tu ne sais pas": "tu sais", "nous ne savons pas": "nous savons", "vous ne savez pas": "vous savez",
        "ils ne savent pas": "ils savent", "je ne ferai pas": "je ferai", "tu ne feras pas": "tu feras", "il ne fera pas": "il fera",
        "nous ne ferons pas": "nous ferons", "vous ne ferez pas": "vous ferez", "ils ne feront pas": "ils feront", "je ne veux pas": "je veux",
        "tu ne veux pas": "tu veux", "il ne veut pas": "il veut", "nous ne voulons pas": "nous voulons", "vous ne voulez pas": "vous voulez"
    },
    "korean": {
        "않아": "안", "안돼": "안되", "도움": "도움", "않습니다": "안습니다", "안해": "안해", 
        "가": "가다", "하다": "해", "하니": "하고", "했을": "했다", "있어요": "있다",
        "가요": "가다", "못해": "못하다", "왔어요": "왔다", "해야": "해야", "행동": "행동하다",
        "갈게": "갈게요", "먹어요": "먹다", "고마워": "고맙다", "가자": "가자", "앉아": "앉다", 
        "이해해": "이해하다", "가질": "갖다", "갈": "가다", "못해요": "못하다", "걸": "걸다",
        "됐어요": "됐다", "주면": "주다", "서": "서다", "있지": "있다", "하세요": "하다",
        "바꿔": "바꾸다", "얘기해": "얘기하다", "하고싶어": "하고 싶다", "하고싶었어": "하고싶었다",
        "하니까": "하니까", "거든요": "거야", "지금": "현재"
    }
}


    def clean_text(self, text):
        """
        Basic text cleaning function to remove numbers, punctuation, and stopwords.

        :param text: Input text to be cleaned.
        :return: Cleaned text.
        """
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        if self.remove_stopwords:
            text = ' '.join(word for word in text.split() if word not in self.stop_words)
        if self.remove_emojis:
            text = self.remove_emojis_from_text(text)
        if self.handle_contractions:
            text = self.expand_contractions(text)
        return text

    def remove_emojis_from_text(self, text):
        """
        Removes emojis from the given text.

        :param text: Input text.
        :return: Text without emojis.
        """
        return emoji.replace_emoji(text, replace='')

    def expand_contractions(self, text):
        """
        Expands contractions (e.g., "don't" -> "do not") based on the language.

        :param text: Input text.
        :return: Text with expanded contractions.
        """
        contractions = self.contractions.get(self.language, {})
        for contraction, full_form in contractions.items():
            text = re.sub(rf"\b{contraction}\b", full_form, text)
        return text

    def advanced_clean(self, text):
        """
        Advanced text cleaning: remove URLs, very small words, and reduce repeated characters.

        :param text: Input text.
        :return: Cleaned text with advanced processing.
        """
        text = self.clean_text(text)
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove very small words
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated characters
        return text.strip()

    def text_normalization(self, text):
        """
        Normalize text by converting numbers to words.

        :param text: Input text.
        :return: Normalized text.
        """
        number_map = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", 
            "6": "six", "7": "seven", "8": "eight", "9": "nine"
        }
        text = re.sub(r'\d', lambda x: number_map[x.group()], text)
        return text

    def remove_special_characters(self, text):
        """
        Remove non-alphabetic and non-numeric characters.

        :param text: Input text.
        :return: Text without special characters.
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def handle_html_tags(self, text):
        """
        Remove HTML tags from the text.

        :param text: Input text.
        :return: Text without HTML tags.
        """
        return re.sub(r'<.*?>', '', text)

    def spell_correction(self, text):
        """
        Correct spelling mistakes in the text.

        :param text: Input text.
        :return: Text with corrected spelling.
        """
        return self.speller.correct(text)

    def preprocess_multi_language(self, text, language='english'):
        """
        Remove stopwords for different languages.

        :param text: Input text.
        :param language: Language for stopword filtering.
        :return: Text without stopwords for the given language.
        """
        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text

    def get_most_common_words(self, text, num_words=10):
        """
        Get the most common words in a text.

        :param text: Input text.
        :param num_words: Number of most common words to return.
        :return: List of most common words and their counts.
        """
        words = text.split()
        word_count = Counter(words)
        return word_count.most_common(num_words)

    def apply_to_dataframe(self, df, column_name, method, *args, **kwargs):
        """
        Apply a text processing method to a specific column in a pandas DataFrame.

        :param df: DataFrame containing the text data.
        :param column_name: Column name to apply the method on.
        :param method: Method to apply (e.g., clean_text, advanced_clean).
        :param args: Additional arguments to pass to the method.
        :param kwargs: Additional keyword arguments to pass to the method.
        :return: DataFrame with processed text.
        """
        df[column_name] = df[column_name].apply(lambda x: getattr(self, method)(x, *args, **kwargs))
        return df



