import spacy
import textacy.extract
from typing import List, Union
from spacy.tokens import Span, Token

def get_span_text(span_or_list: Union[Span, List[Span]]) -> str:
    """
    Robustly converts a spaCy Span or a list of Spans (for compound
    subjects/objects) into a single, clean string.
    """
    if isinstance(span_or_list, list):
        # This handles compound subjects/objects, e.g., "The cat and the dog"
        # We join them with "and" to maintain semantic meaning.
        return " and ".join(span.text.strip() for span in span_or_list)
    else:
        # This is a single span
        return span_or_list.text.strip()

def get_verb_lemma(verb_list: List[Token]) -> str:
    """
    Converts a list of verb tokens into a single lemmatized string.
    e.g., [was, awarded] -> "award"
    e.g., [is, stuck] -> "stick" (lemma of 'stuck')
    """
    # We find the *last* verb/aux token in the list and use its lemma.
    # This correctly handles "was awarded" (lemma of 'awarded' is 'award')
    # and "is stuck" (lemma of 'stuck' is 'stick').
    
    # Find the main verb (non-auxiliary)
    main_verb = None
    for token in reversed(verb_list):
        if token.pos_ == "VERB":
            main_verb = token
            break
            
    if main_verb:
        return main_verb.lemma_
    else:
        # If no main verb (e.g., "is"), just use the lemma of the last token.
        return verb_list[-1].lemma_


# --- Test ---
print("Loading spaCy model (en_core_web_sm)...")
# We need the full pipeline for textacy: tagger, parser, and NER
nlp = spacy.load("en_core_web_sm") 

# This is the full list of test cases to ensure robustness
test_sentences = [
    # 1. Active Voice (Simple)
    "Wilhelm Röntgen received the first Nobel Prize in Physics.",
    # 2. Passive Voice (Preposition)
    "The first Nobel Prize in Physics was awarded to Wilhelm Röntgen.",
    # 3. Passive Voice (Agent)
    "The mouse was chased by the cat.",
    # 4. Compound Subject (This caused the 'list' object error)
    "The cat and the dog chased the mouse.",
    # 5. Compound Object
    "The agent finds entities and extracts relations.",
    # 6. Attributive ("is a..." pattern)
    "This is a simple sentence.",
    # 7. Long Phrases
    "The fast, powerful B200 GPU processed the 14GB file.",
    # 8. Dative (double object)
    "She gave the book to him.",
    # 9. Complex Verb
    "The script is stuck at the NetworkX warning.",
    # 10. Negative Cases (Should produce [])
    "Wow, that's fast!",
    "A sentence with no verb.",
    "Running the build script."
]

print("Running Textacy OIE test...")

for text in test_sentences:
    doc = nlp(text)
    
    # textacy's SVO function is a generator
    triples_generator = textacy.extract.subject_verb_object_triples(doc)
    
    triples = []
    for s, v, o in triples_generator:
        # Use our robust helper functions to convert
        subject_text = get_span_text(s)
        verb_text = get_verb_lemma(v)
        object_text = get_span_text(o)
        
        triples.append((subject_text, verb_text, object_text))

    print(f"\n--- TEXT ---")
    print(text)
    print(f"TRIPLES: {triples}")