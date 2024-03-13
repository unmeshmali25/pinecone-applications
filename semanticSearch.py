import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from DLAIUtils import Utils
import DLAIUtils
import json
import os
import time
import torch
from tqdm.auto import tqdm


dataset = load_dataset('quora', split='train[240000:290000]')
print(f"dataset downloaded successfully..")

questions = []
for record in dataset['questions']:
    questions.extend(record['text'])
question = list(set(questions))
print('\n'.join(questions[:10]))
print('-' * 50)
print(f'Number of questions: {len(questions)}')


# Check CUDA and setup the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Query adn encoding using the all-MiniLM-L6-v2 model
query = 'which city is the most populated in the world?'
xq = model.encode(query)
xq.shape

# Steup Pinecone
utils = Utils()

with open('./pineapi.json', 'r') as file:
	data = json.load(file)
os.environ['PINECONE_API_KEY'] = data['PINECONE_API_KEY']
PINECONE_API_KEY = data['PINECONE_API_KEY']


pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
print(INDEX_NAME)
pinecone.create_index(name=INDEX_NAME, 
    dimension=model.get_sentence_embedding_dimension(), 
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-west-2'))

index = pinecone.Index(INDEX_NAME)
print(index)