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
        # Access the Pinecone API key from environment variables
        PineconeAPIKey = os.environ.get("PINECONE_API_KEY")
        
        # 1. We will load our embedding model (for turning the query into a vector)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 2. We will initialize our connection to Pinecone (THE CORRECT V3+ WAY)
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")        
        # This is the new way to initialize the main client
        self.pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)
        # This is how you connect to your specific index, by its name
        self.pinecone_index = self.pinecone_client.Index("aws-rag")
        
        # 3. We will initialize our connection to AWS Bedrock
        # self.bedrock_client = boto3.client("bedrock-runtime", "us-west-2")
        
        print("NexusFlowService is initialized and ready.")


    # This is the "sticker" that tells BentoML this is a "Front Door" (API)
    # It will expect a POST request with JSON: {"query": "some question"}
    @bentoml.api
    def answer_question(self, query: str) -> dict:
        
        # This is the main RAG logic for our "Front Door"

        # STEP 1: RETRIEVE (Get documents from Pinecone)
        # First, turn the user's text 'query' into a vector
        # query_vector = self.embedding_model.encode(query)

        # Then, search Pinecone with that vector to get relevant text chunks
        # search_response = self.pinecone_index.query(vector=query_vector.tolist(), top_k=3, include_metadata=True)
        # context_chunks = [match['metadata']['text'] for match in search_response['matches']]

        # STEP 2: AUGMENT (Build the prompt for the LLM)
        # We build a prompt using the user's query and the chunks we found
        # prompt = f"""
        # User question: {query}
        #
        # Relevant documents:
        # {" ".join(context_chunks)}
        #
        # Please answer the user's question based *only* on the documents provided.
        # """

        # STEP 3: GENERATE (Get the answer from Bedrock)
        # We send the big prompt to the LLM (Bedrock)
        # llm_response = self.bedrock_client.invoke_model(body=prompt, ...)
        # final_answer = llm_response['body'].read()
        
        # For now, just return a placeholder
        final_answer = "This is a placeholder answer."
        context_chunks = "These are placeholder chunks."

        # We return a dictionary, which BentoML turns into JSON
        return {
            "answer": final_answer,
            "sources": context_chunks
        }
