## Prompt 2 — Baseline RAG (Phase 1, part A)

Build a BASELINE (non-agentic) RAG pipeline over the extracted pages-4-27     
  content,                                                                      
  as a notebook called baseline_rag.ipynb :                                     
  - Chunk the extracted content. For now use a simple, deliberately naive       
  chunking                                                                      
    strategy (fixed-size or simple recursive split) — we WANT to expose where   
  naive                                                                         
    chunking breaks tables, so don't make it clever yet.                        
  - Embed with Gemini embeddings, persist to chroma_db/ , retrieve top-k (k=4). 
  - Generate a grounded answer with gpt-4o-mini, temperature 0, using only      
  retrieved                                                                     
    context.                                                                    
  - Add a clearly-labelled markdown cell explaining this is the baseline and    
  its                                                                           
    known structural weakness (no way to verify retrieval, no recovery,         
  one-shot).                                                                    
                                                                                
  Don't add grading, routing, rewriting, or web search — that's deliberately    
  absent. 