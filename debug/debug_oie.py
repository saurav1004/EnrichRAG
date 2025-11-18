import spacy
from spacy.language import Language
from collections import defaultdict

# --- This is the FINAL OIE FUNCTION (v12) ---
SUBJ_DEPS = {"nsubj", "nsubjpass"} # nsubjpass is for passive
OBJ_DEPS = {"dobj", "pobj", "attr"}

@Language.component("robust_oie_v12")
def extract_oie_triples_v12(doc):
    triples = []
    
    # Use defaultdict to group tokens by their head (the verb/relation)
    relations = defaultdict(lambda: {"subjects": [], "objects": []})

    for token in doc:
        # --- 1. Find all Subjects ---
        if token.dep_ in SUBJ_DEPS:
            # token.head is the verb (e.g., 'received' or 'awarded')
            relation_verb = token.head
            subject_span = doc[token.left_edge.i : token.right_edge.i + 1].text
            relations[relation_verb]["subjects"].append(subject_span)

        # --- 2. Find all Objects ---
        
        # Case 1: Direct Object (dobj) or Attribute (attr)
        if token.dep_ == "dobj" or token.dep_ == "attr":
            relation_verb = token.head
            object_span = doc[token.left_edge.i : token.right_edge.i + 1].text
            relations[relation_verb]["objects"].append(object_span)
        
        # Case 2: Prepositional/Dative Object (pobj)
        # e.g., "awarded -> to -> Röntgen" OR "chased -> by -> cat"
        if token.dep_ in OBJ_DEPS and token.head.dep_ in ("prep", "dative", "agent"):
            # The relation_verb is the head of the preposition
            # (Röntgen -> to -> awarded)
            relation_verb = token.head.head
            object_span = doc[token.left_edge.i : token.right_edge.i + 1].text
            relations[relation_verb]["objects"].append(object_span)

    # --- 3. Combine the results ---
    for relation_verb, s_and_o in relations.items():
        if s_and_o["subjects"] and s_and_o["objects"]:
            
            # Use the verb's lemma (base form) if available,
            # otherwise, just use its text.
            relation = relation_verb.lemma_ if relation_verb.lemma_ else relation_verb.text
            
            for s in s_and_o["subjects"]:
                for o in s_and_o["objects"]:
                    triples.append((s.strip(), relation.strip(), o.strip()))
    
    doc.set_extension("oie_triples", default=[], force=True)
    doc._.oie_triples = triples
    return doc
# --- End of New OIE function ---

# --- Test ---
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("robust_oie_v12", after="parser")

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


print("Running OIE test...")
for text in test_sentences:
    doc = nlp(text)
    print(f"\n--- TEXT ---")
    print(text)
    print(f"TRIPLES: {doc._.oie_triples}")