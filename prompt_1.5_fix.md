## Prompt 1.5 - Fix venv

The venv wasn't actually created in Prompt 0. Create it now and install       
  requirements:                                                                 
                                                                                
    python3 -m venv .venv                                                       
    source .venv/bin/activate                                                   
    pip install -r requirements.txt                                             
                                                                                
  Then confirm the install worked by printing the installed versions of         
  pdfplumber, langchain, langgraph, and chromadb. If requirements.txt is        
  missing or incomplete, show it to me first so I can check it before           
  installing.                                                                   
                                                                                
  Separately, before we go further: move the real API keys out of .env.example  
  into .env, restore .env.example to placeholder values only, and confirm .env  
  is git-ignored. I'm going to rotate those keys regardless. 