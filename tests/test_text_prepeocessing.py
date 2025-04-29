import pytest
import sys 
import os
sys.path.append(os.path.abspath(os.path.dirname("__file__")))
from OutText_preprocessing.text_preprocessing import TextPreprocessor
@pytest.fixture
def sample_text():
    return "This is a SAMPLE sentence, with punctuation! And stopwords?"

def test_lowercase_only(sample_text):
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False, remove_stopwords=False, lemmatize=False)
    processed = preprocessor.transform([sample_text])
    assert processed[0].islower()

def test_remove_punctuation(sample_text):
    preprocessor = TextPreprocessor(lowercase=False, remove_punctuation=True, remove_stopwords=False, lemmatize=False)
    processed = preprocessor.transform([sample_text])
    assert "," not in processed[0] and "!" not in processed[0] and "?" not in processed[0]

def test_remove_stopwords(sample_text):
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True, remove_stopwords=True, lemmatize=False)
    processed = preprocessor.transform([sample_text])
    assert "this" not in processed[0] and "is" not in processed[0] and "and" not in processed[0]

def test_lemmatization():
    text = "The cats are running in the gardens"
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
    processed = preprocessor.transform([text])
    assert "cat" in processed[0] and "run" in processed[0] and "garden" in processed[0]

def test_multiple_texts():
    texts = [
        "Dogs are playing!", 
        "He was walking to the stores.", 
        "Birds fly HIGH above."
    ]
    preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True, remove_stopwords=True, lemmatize=True)
    processed = preprocessor.transform(texts)
    assert isinstance(processed, list)
    assert all(isinstance(p, str) for p in processed)
    assert len(processed) == 3

def test_empty_string():
    preprocessor = TextPreprocessor()
    processed = preprocessor.transform([""])
    assert processed[0] == ""

def test_invalid_input():
    preprocessor = TextPreprocessor()
    with pytest.raises(TypeError):
        preprocessor.transform("This is not a list")  
