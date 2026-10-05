#import modules
import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

 
load_dotenv()
client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

#pdf reader
pdf_path="documents/sample.pdf"
reader=PdfReader(pdf_path)
text=""

for page in reader.pages:
    text += page.extract_text()

#divide pdf into chunks
chunk_size=500
chunks=[]

for i in range(0,len(text),chunk_size):
    chunk=text[i:i+chunk_size]
    chunks.append(chunk)

#chromadb
chroma_client=chromadb.PersistentClient(
    path="./chroma_db"
)
collection=chroma_client.get_or_create_collection(
    name="chunks"
)

#chunk embedding
document_vector=[]
for document in chunks: 
    Response=client.models.embed_content(
      model="gemini-embedding-2",
      contents=document
)
    #convert it into vector
    vector=Response.embeddings[0].values

    document_vector.append(vector)

# store it into chromadb
for i,(document,vector) in enumerate(zip(
    chunks,document_vector
)):
    collection.add(ids=[str(i)],
                   embeddings=[vector],
                   documents=[document])


question=input("Ask your question: ")

#question embedding
question_response=client.models.embed_content(
    model="gemini-embedding-2",
    contents=question
)


#convert it into vector
question_vector=question_response.embeddings[0].values

#similarity search
results= collection.query(query_embeddings=[question_vector],
                          n_results=2)
best_document=results["documents"][0]

print("Relevent chunks:")
for doc in best_document:
    print(doc)

#chunks in context
context="\n\n".join(best_document)


#send retrieve chunk to LLM

prompt=f"""
Answer the question using only the context below.
context:
{context}

question
{question}
"""

#generate final answer
response=client.models.generate_content(
    model='gemini-3.5-flash-lite',
    contents=prompt
)

print("\nFinal Answer:")
print(response.text)




