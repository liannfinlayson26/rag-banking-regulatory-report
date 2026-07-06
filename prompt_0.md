## Prompt 0 — Project scaffold
Set up a fresh Python project for a retrieval-augmented generation system     
  over a                                                                        
  banking regulatory document. Create:                                          
                                                                                
  - A venv and a requirements.txt pinning: langchain, langchain-openai,         
    langchain-google-genai, langchain-chroma, langgraph, chromadb,              
  tavily-python,                                                                
    python-dotenv, pydantic, and a TABLE-AWARE pdf parser (use pdfplumber, and  
  add                                                                           
    camelot-py or unstructured if you judge it better for bordered tables —     
  explain                                                                       
    your choice).                                                               
  - A .env.example with OPENAI_API_KEY, GOOGLE_API_KEY, TAVILY_API_KEY          
  placeholders,                                                                 
    and a .gitignore that excludes .env, venv/, chroma_db/, and the downloaded  
  PDF.                                                                          
  - A README stub and a docs/ folder for dated build logs.                      
  - A data/ folder.                                                             
                                                                                
  Do not write any RAG code yet. Just the scaffold. Show me the structure when  
  done. 