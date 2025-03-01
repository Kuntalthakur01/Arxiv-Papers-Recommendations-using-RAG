import os
import re
import json

# directory paths (you guys need to change since this is mine)
latex_dir = '/Users/karthikdubba/Classes/Fall 2024/CSE 511/Project/ArxivLatex'
output_dir = '/Users/karthikdubba/Classes/Fall 2024/CSE 511/Project/equation_json'
os.makedirs(output_dir, exist_ok=True)

# latex math eq
equation_patterns = [
    r'\\begin{equation}(.*?)\\end{equation}',  
    r'\$\$(.*?)\$\$',                         
    r'\$(.*?)\$',                              
    r'\\\[(.*?)\\\]'                           
]

# min length for valid equations
MIN_EQUATION_LENGTH = 3 

def clean_equation(eq):
    # remove wrapping math delimiters
    eq = re.sub(r'\\begin{equation}|\\end{equation}', '', eq) 
    eq = re.sub(r'\\\[|\\\]', '', eq)                          
    eq = re.sub(r'\$\$|\$', '', eq)                            
    return eq.strip()

# extract equations from LaTeX content
def extract_equations(latex_content):
    equations = []
    for pattern in equation_patterns:
        matches = re.findall(pattern, latex_content, re.DOTALL)
        equations.extend([clean_equation(eq) for eq in matches])
    # filter out short equations, sometimes it extracts one character
    return [eq for eq in equations if len(eq) >= MIN_EQUATION_LENGTH]

# go through LaTeX files
for file_name in os.listdir(latex_dir):
    if file_name.endswith(".tex"):
        file_path = os.path.join(latex_dir, file_name)
        
        # try opening the file with UTF-8 encoding; fallback to ISO-8859-1
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                latex_content = f.read()
        except UnicodeDecodeError:
            print(f"UTF-8 decoding failed for {file_name}. Retrying with ISO-8859-1...")
            with open(file_path, "r", encoding="ISO-8859-1") as f:
                latex_content = f.read()
        
        # extract equations
        equations = extract_equations(latex_content)
        
        # sequential numbering
        equations_json = {}
        for idx, eq in enumerate(equations):
            eq_key = f"eq-{idx + 1}" 
            equations_json[eq_key] = eq  # Save cleaned equation

        # save equations to a JSON file
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(equations_json, f, indent=4, ensure_ascii=False)

        print(f"Extracted equations saved to {output_file}")
