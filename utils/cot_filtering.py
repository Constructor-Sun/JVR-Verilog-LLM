#!/usr/bin/env python3
import json
import re
import os

def extract_cot_and_code(response):
    """
    Extracts the Chain of Thought (CoT) and Verilog code from the response.
    CoT is inside <think>...</think>.
    Verilog code is after </think> and enclosed in ```verilog ... ```.
    """
    # Extract CoT
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response, re.DOTALL)
    cot = think_match.group(1).strip() if think_match else ""

    # Extract Code
    # Search for code after </think> if possible, otherwise search entire string
    if "</think>" in response:
        after_think = response.split("</think>")[-1]
    else:
        after_think = response

    code_pattern = r"```verilog(.*?)```"
    code_match = re.search(code_pattern, after_think, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""
    
    return cot, code

def extract_verilog_entities(code):
    """
    Extracts potential identifiers (entities) from Verilog code.
    Excludes common Verilog keywords.
    """
    verilog_keywords = {
        "module", "endmodule", "input", "output", "inout", "wire", "reg", "always", "begin", "end",
        "if", "else", "case", "endcase", "default", "parameter", "assign", "posedge", "negedge",
        "or", "and", "not", "xor", "nand", "nor", "xnor", "initial", "integer", "genvar", "generate",
        "endgenerate", "for", "while", "repeat", "forever", "function", "endfunction", "task", "endtask",
        "logic", "bit", "byte", "shortint", "int", "longint", "time", "real", "shortreal", "chandle",
        "string", "event", "clocking", "endclocking", "interface", "endinterface", "modport",
        "property", "endproperty", "sequence", "endsequence", "assert", "cover", "assume"
    }
    
    # Remove comments
    code_no_comments = re.sub(r"//.*", "", code)
    code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)
    
    # Extract words
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code_no_comments)
    
    # Filter keywords and short/common irrelevant words
    entities = set()
    for w in words:
        if w not in verilog_keywords and len(w) > 1:
            entities.add(w)
            
    return entities

def calculate_metrics(cot, code, entities):
    """
    Calculates quality metrics for the CoT based on the code.
    """
    if not cot:
        return None
    
    cot_lower = cot.lower()
    
    # 1. Entity Matching
    # Check how many of the code entities are mentioned in the CoT
    if not entities:
        entity_matching_score = 0.0
    else:
        # We check case-insensitive match
        entities_found = 0
        for entity in entities:
            if entity.lower() in cot_lower:
                entities_found += 1
        entity_matching_score = entities_found / len(entities)
    
    # 2. Reasoning Keyword Density
    reasoning_keywords = [
        "because", "therefore", "thus", "hence", "so", "implies", "since", "reason", 
        "first", "then", "finally", "assume", "check", "verify", "ensure", "logic",
        "design", "implement", "consider", "step", "if", "when", "must", "should", "need"
    ]
    
    cot_words = re.findall(r"\b\w+\b", cot_lower)
    cot_word_count = len(cot_words)
    
    if cot_word_count == 0:
        reasoning_density = 0.0
    else:
        reasoning_count = sum(1 for w in cot_words if w in reasoning_keywords)
        reasoning_density = reasoning_count / cot_word_count
    
    # 3. Structural Analysis
    # Check for steps (numbered lists), code blocks, paragraphs
    has_numbered_list = bool(re.search(r"^\s*\d+\.", cot, re.MULTILINE))
    has_steps_keyword = "step" in cot_lower
    num_newlines = cot.count('\n')
    # Normalized structure score (simple heuristic)
    structure_score = 0.0
    if has_numbered_list: structure_score += 0.5
    if has_steps_keyword: structure_score += 0.2
    if num_newlines > 5: structure_score += 0.3
    
    # 4. Information Gain / Content Ratio
    code_len = len(code)
    cot_len = len(cot)
    # Ratio of CoT length to Code length. 
    # Too short might be bad. Too long might be verbose but usually good for CoT.
    info_gain = cot_len / code_len if code_len > 0 else 0.0
    
    return {
        "cot_length": cot_len,
        "code_length": code_len,
        "entity_matching": entity_matching_score,
        "reasoning_density": reasoning_density,
        "structure_score": min(structure_score, 1.0),
        "info_gain": info_gain
    }

def main():
    input_file = "./data/training_data/integral/codev_r1_sft_think_filtered.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return

    print(f"Processing {input_file}...")
    print(f"{'ID':<5} | {'CoT Len':<8} | {'Code Len':<8} | {'Entity Match':<12} | {'Reasoning':<10} | {'Structure':<10} | {'Info Gain':<10}")
    print("-" * 85)

    processed_count = 0
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if processed_count >= 100:
                break
            
            try:
                data = json.loads(line)
                response = data.get("response", "")
                
                cot, code = extract_cot_and_code(response)
                
                if not cot or not code:
                    # Skip if missing CoT or Code, but user said "check data", 
                    # maybe we should report them as invalid? 
                    # For now, let's just log them and not count them towards the 100 "results" 
                    # if we consider "results" as valid analyses.
                    # Or maybe we count them but show zeros.
                    # Let's count them to be safe and show they are empty.
                    pass 

                entities = extract_verilog_entities(code)
                metrics = calculate_metrics(cot, code, entities)
                
                if metrics:
                    print(f"{i:<5} | {metrics['cot_length']:<8} | {metrics['code_length']:<8} | "
                          f"{metrics['entity_matching']:<12.4f} | {metrics['reasoning_density']:<10.4f} | "
                          f"{metrics['structure_score']:<10.4f} | {metrics['info_gain']:<10.4f}")
                    
                    # if metrics['info_gain'] > 60:
                    #     print("\n" + "="*80)
                    #     print(f"HIGH INFO GAIN DETECTED (ID: {i}, Gain: {metrics['info_gain']:.2f})")
                    #     print("="*80)
                    #     print("--- CoT ---")
                    #     print(cot)
                    #     print("\n--- Code ---")
                    #     print(code)
                    #     print("="*80 + "\n")

                    processed_count += 1
                else:
                     print(f"{i:<5} | {'MISSING CoT':<30}")
                     processed_count += 1

            except json.JSONDecodeError:
                print(f"{i:<5} | JSON ERROR")
                processed_count += 1
                continue

if __name__ == "__main__":
    main()
