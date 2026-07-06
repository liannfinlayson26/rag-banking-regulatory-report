## Prompt 3 — Stress test the baseline (Phase 1, part B)

Add a stress-test section to baseline_rag.ipynb that runs 10 questions        
  through the                                                                   
  EXISTING baseline pipeline. Reuse the already-built chroma_db/ collection     
  "pillar3_baseline" as-is — do NOT re-chunk, re-embed, or rebuild the store,   
  and do                                                                        
  NOT add any grading/routing/rewriting/web search. This is the naive baseline  
  being                                                                         
  tested on purpose.                                                            
                                                                                
  For each question, ask ONLY the bare question (never show the model the gold  
  answer)                                                                       
  and capture the baseline's answer. I've given gold answers + sources so we    
  can score                                                                     
  afterwards — verify each gold against data/extracted/ and FLAG any mismatch   
  rather                                                                        
  than overwriting. Note: RWA cells are in $m (Total = 888,647 $m = 888.6 $bn); 
  treat a right figure with the wrong unit as a distinct failure.               
                                                                                
  After running, print a scoring table:                                         
  question | baseline answer | gold | grounded? | numerically correct? |        
  complete? | trap                                                              
                                                                                
  Questions:                                                                    
                                                                                
  1. "What is HSBC's CET1 capital ratio as at 31 December 2025?"                
     GOLD: 14.9% (unchanged from 31 Dec 2024). [Highlights p4 / Table 1 p5]     
     TRAP: number-conflation — five quarter-end ratios in one row               
  (14.9/14.5/14.6/14.7/14.9).                                                   
                                                                                
  2. "What was HSBC's CET1 capital (the $bn figure, not the ratio) at 31 Dec    
  2025?"                                                                        
     GOLD: $132.6bn. [p4 / Table 1 row 1]                                       
     TRAP: returns the ratio, or a prior-period capital value.                  
                                                                                
  3. "What is HSBC's leverage ratio, and did it rise or fall over 2025?"        
     GOLD: 5.3%, DOWN from 5.6% at 31 Dec 2024. [Highlights p4]                 
     TRAP: temporal — "down from" may yield 5.6 as current.                     
                                                                                
  4. "What is the combined RWA for market risk and counterparty credit risk at  
  31 Dec 2025?"                                                                 
     GOLD: 38,490 + 42,380 = 80,870 $m (= $80.9bn). [OV1 / Table 7,             
  by-risk-type table p4]                                                        
     TRAP: aggregation — returns one row, or Total (888,647), or fabricates.    
                                                                                
  5. "Is HSBC's reported CET1 ratio on a transitional or end-point basis?"      
     GOLD: Both the same — IFRS 9 transitional ended 1 Jan 2025, CRR II         
  grandfathering                                                                
     ended 28 Jun 2025, so capital figures are identical on both bases. [Key    
  Metrics preamble p4]                                                          
     TRAP: basis/qualifier — ignores the caveat, just restates a number.        
                                                                                
  6. "What were HSBC's LCR and NSFR at 31 Dec 2025, and what is the HQLA        
  amount?"                                                                      
     GOLD: LCR 137%, NSFR 143%, average HQLA $702bn. [Liquidity highlights p4]  
     TRAP: compound — drops one of the three parts silently.                    
                                                                                
  7. "What net CET1 impact did the Hang Seng Bank privatisation have, and       
  when?"                                                                        
     GOLD: net 110 bps in January 2026 (day-one ~120 bps, partly offset by ~10  
  bps),                                                                         
     based on the 31 Dec 2025 ratio. [Highlights p4]                            
     TRAP: conflates the 110/120/10 bps figures.                                
                                                                                
  8. "What total capital charge does HSBC apply as a percentage of RWAs, and    
  under which regulation?"                                                      
     GOLD: 8% of RWAs, set by Article 92(1) of CRR II. [Comparatives &          
  references p3]                                                                
     TRAP: cross-reference — returns 8% without the Article reference.          
                                                                                
  9. "What is HSBC Holdings' total RWA, and in what currency?"                  
     GOLD: $888,647m (= $888.6bn), US dollars. [p4 + currency definition p1]    
     TRAP: units — omits USD, or gives $bn without noting source cells are $m.  
                                                                                
  10. "What is the operational risk RWA, and how much did it change from 2024?" 
      GOLD: 120,716 $m vs 106,472 $m = increase of 14,244 $m (≈ +$14.2bn).      
  [by-risk-type table p4]                                                       
      TRAP: aggregation/delta — returns one year, or miscomputes.               
                                                                                
  Then write docs/stress-test-findings.md summarising which traps fired, with   
  the                                                                           
  actual baseline answers as evidence. Be honest — if baseline got one right,   
  say so,                                                                       
  and note any case where the right page was retrieved but the answer still     
  failed                                                                        
  (that isolates chunking failures from retrieval failures). 
