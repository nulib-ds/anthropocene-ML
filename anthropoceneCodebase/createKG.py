import pandas as pd
from transformers import pipeline

# Load the CSV file
csv_file = '/home/ysc4337/aerith/anthropocene-reconcile/anthropocene-data/results/image_captioning_extending_output_length.csv'
df = pd.read_csv(csv_file)

# Initialize the triplet extractor pipeline
triplet_extractor = pipeline('text2text-generation', model='Babelscape/rebel-large', tokenizer='Babelscape/rebel-large')

# Function to extract triplets from a given text
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

# Extract triplets for each caption in the CSV
for caption in df['caption']:
    extracted_text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(caption, return_tensors=True, return_text=False, max_length=28000)[0]["generated_token_ids"]])
    extracted_triplets = extract_triplets(extracted_text[0])
    print(extracted_triplets)
