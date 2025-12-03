import json
import bentoml
import boto3
import pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer



# This is the "sticker" that tells BentoML this class is the main "House"
@bentoml.service
class NexusFlowService:

    def __init__(self):
        
        # Load environment variables from .env file
        load_dotenv()
        
        # 1. We will load our embedding model (for turning the query into a vector)
        embedding_model_name =  os.environ.get("EMBEDDING_MODEL_NAME")
        self.embedding_model = SentenceTransformer(embedding_model_name)  

        # 2. We will initialize our connection to Pinecone (keys in .env)
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")        
        self.pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)
        self.pinecone_index = self.pinecone_client.Index("aws-rag")
        
        # 3. We will initialize our connection to AWS Bedrock
        # 3A. This cost money so we will wait until everything else is working
        # self.bedrock_client = boto3.client("bedrock-runtime", "us-west-2")
        
        print("NexusFlowService is initialized and ready.")

    # This is the "sticker" that tells BentoML this is an API
    # It will expect a POST request with JSON: {"query": "some question"}
    @bentoml.api
    def answer_question(self, query: str) -> dict:
        # This is the main RAG logic
        # STEP 1: RETRIEVE (Get documents from Pinecone)
        # First, turn the user's text 'query' into a vector
        query_vector = self.embedding_model.encode(query)

        # Then, search Pinecone with that vector to get relevant text chunks
        search_response = self.pinecone_index.query(vector=query_vector.tolist(), top_k=3, include_metadata=True)
        context_chunks = [match['metadata']['text'] for match in search_response['matches']]

        # STEP 2: AUGMENT (Build the prompt for the LLM)
        # Now we have the query, and context_chunks, pass it to bedrock to have our selected LLM answer it
        # Initial prompt just needs to work and show we are NOT using the training knowledge, but the retrieved docs
        #prompt = f"""
        #    User question: {query}
        #    Relevant documents: {" ".join(context_chunks)}
        #    Please answer the user's question based *only* on the relevant documents.
        #    If the answer is contained within the documents, provide a concise and accurate answer in your own words
        #    followed by a direct quote from the document as a source.
        #    If you dont feel the provided documents are sufficient to answer the question, please say "I don't know."
        #    Do not rely on prior knowledge or make up an answer, anything outside of the relevant documents should be ignored.
        #"""
        prompt = f"""Tell me your favorite food and repeat {query}"""
        # STEP 3: GENERATE (Get the answer from Bedrock)
        # We send the big prompt to the LLM (Bedrock)
        llm_response = self.bedrock_client.invoke_model(
            modelId='amazon.nova-2-lite-v1:0',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7
            })
        )
        # Read the response body, decode it, and parse the JSON
        response_body = json.loads(llm_response['body'].read().decode('utf-8'))
        final_answer = response_body.get('completion', response_body.get('output', str(response_body)))        
        # For now, just return a placeholder
        #final_answer = "This is a placeholder answer."
        #context_chunks = "These are placeholder chunks."

        # We return a dictionary, which BentoML turns into JSON
        return {
            "answer": final_answer,
            "sources": context_chunks
        }
