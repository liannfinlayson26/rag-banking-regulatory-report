## Prompt 3.5: eval_set.json 

Update eval_set.json (and any stress-test doc) so every gold answer's cited   
  page                                                                          
  matches the actual PDF page, not the extracted-file index — you noted a 1–2   
  page                                                                          
  offset, Q9's currency on p20/27, and Q8's Article on p4. I want citations a   
  reader                                                                        
  can verify by opening that exact page.

## Prompt 3.6: eval-set.md
Generate a human-readable markdown version of eval_set.json as                
  docs/eval-set.md —                                                            
  a table with columns: ID | Question | Gold | Trap | PDF page(s) | Where. Keep 
  eval_set.json as the canonical machine-readable source; the markdown is just  
  a                                                                             
  readable mirror for review and the README. If they ever diverge, JSON wins.