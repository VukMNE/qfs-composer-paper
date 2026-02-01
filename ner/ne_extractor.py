import spacy_udpipe
import spacy
import logging


def load_spacy_model(language):
    if language == "sl":
        logging.info("Loading slovenian from spacy_udpipe")
        spacy_udpipe.download("sl")
        return spacy_udpipe.load("sl")
    return spacy.load(language)

def get_noun_chunks(doc):
    """
    Manually extracts noun chunks for Slovenian using POS tagging and dependency parsing.
    """
    noun_chunks = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # Targeting nouns and proper nouns
            chunk = token.text

            # Include adjectives or determiners before the noun
            for left in token.lefts:
                if left.pos_ in ["ADJ", "DET", "NUM"]:
                    chunk = left.text + " " + chunk

            noun_chunks.append(chunk)

    return noun_chunks

def get_named_entities(doc):
    """
    Manually extracts named entities for Slovenian using POS tagging and dependency parsing.
    """
    named_entities = []
    entity = []

    for token in doc:
        # Check if the token is a proper noun (PROPN) or noun (NOUN)
        if token.pos_ in ["PROPN", "NOUN"]:
            if not entity:  # Start a new entity
                entity_label = "ORG" if token.dep_ == "nsubj" else "MISC"  # Approximate categories
            entity.append(token.text)
        else:
            if entity:  # If we have collected an entity, store it
                named_entities.append(" ".join(entity))
                entity = []

    # Capture any remaining entity at the end of the loop
    if entity:
        named_entities.append(" ".join(entity))

    print("DEBUG - Extracted named entities:", named_entities)  # Debugging
    return named_entities



class NE_Extractor:
    def __init__(self, language, model=None):
        self.model = model
        if language == 'en':
            try:
                self.spacy_pipeline = spacy.load('en_core_web_sm')
            except OSError:
                logging.warning("Downloading language model for the spaCy model.")
                from spacy.cli import download
                download('en_core_web_sm')
                self.spacy_pipeline = spacy.load('en_core_web_sm')

        elif language == 'sl':
            try:
                logging.info("Attempting to load SLovenian language from spacy_udpipe")
                self.spacy_pipeline = load_spacy_model(language)
            except OSError:
                logging.warning("Downloading Slovenian language model for the spaCy model.")
                from spacy.cli import download
                download('sl_core_news_sm')
                self.spacy_pipeline = spacy.load('sl_core_news_sm')


    def extract_entities(self, text):
        # Dummy implementation for named entity extraction
        doc = self.spacy_pipeline(text)
        if self.spacy_pipeline.lang == "sl":
            #entities = get_named_entities(doc)
            entities = get_noun_chunks(doc)
            #entities.extend(noun_chunks)
            print("DEBUG - Extracted entities FINAL:", list(set(entities)))  # Debugging
            return list(set(entities))  # Remove duplicates        