import re

def clean_text(text: str) -> str:
    """Clean tweet text for training & prediction."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # URLs
    text = re.sub(r'@\w+', ' ', text)               # mentions
    text = re.sub(r'#', '', text)                   # keep word, drop '#'
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # non-alphanumeric -> space
    text = re.sub(r'\s+', ' ', text).strip()
    return text