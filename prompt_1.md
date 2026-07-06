## Prompt 1 — Corpus acquisition + table-aware extraction (Phase 0)

Download the HSBC Pillar 3 PDF from this URL into data/ :                     
  https://www.hsbc.com/-/files/hsbc/investors/hsbc-results/2025/annual/pdfs/hsb 
  c-holdings-plc/260225-pillar-3-disclosures-at-31-december-2025.pdf            
                                                                                
  Then write an extraction script that processes ONLY pages 4–27 (the           
  Highlights,                                                                   
  Key Metrics, Own Funds, Leverage, and Liquidity sections). The hard           
  requirement:                                                                  
  tables must survive extraction with their structure intact. Specifically:     
  - Each table's caption/number, its column headers (including the reporting    
  dates                                                                         
    like "31 Dec 2025" / "31 Dec 2024"), its units row ($bn, %), and its row    
  labels                                                                        
    must stay attached to the correct numeric cells.                            
  - Convert each detected table to clean markdown (or a structured dict) so a   
  row                                                                           
    label and a date column unambiguously map to one value.                     
  - Keep narrative prose separate from tables.                                  
                                                                                
  Output the result to data/extracted/ as one file per page or per section.     
  Then                                                                          
  print, for my review, the extracted version of:                               
    (a) the "RWAs by risk type" table (Credit/CCR/Market/Operational/Total),    
  and                                                                           
    (b) Table 1 "Key metrics (KM1)" — the CET1 row across the five dates.       
  I need to eyeball that the columns didn't scramble before we build anything   
  on top.    