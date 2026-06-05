import sys
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_svg.py <prompt> [output_path] [model_id]")
        sys.exit(1)
        
    prompt_subject = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.svg"
    model_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    base_svg_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    base_svg_content = ""
    if base_svg_path and os.path.exists(base_svg_path):
        try:
            with open(base_svg_path, "r") as f:
                base_svg_content = f.read().strip()
            print(f"Loaded base SVG template from {base_svg_path}")
        except Exception as e:
            print(f"Error reading base SVG: {e}")

    print(f"Loading model and tokenizer: {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print(f"Loaded in {time.time() - t0:.2f}s")
    
    system_prompt = (
        "You are an expert SVG designer. You generate only valid, raw, and high-quality SVG code.\n"
        "Do not wrap the output in markdown code blocks. Start directly with '<svg' and end with '</svg>'.\n"
        "Generate a simple, clean, minimalist vector SVG of the requested subject.\n"
        "You MUST represent the subject accurately using stylized bezier paths (<path d=\"...\">). Do not simply output a circle or square.\n"
        "\n"
        "Example of a waterdrop SVG:\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"100\" height=\"100\">\n"
        "  <path d=\"M50,15 C50,15 80,60 80,70 A30,30 0 1,1 20,70 C20,60 50,15 50,15 Z\" fill=\"#3498db\"/>\n"
        "</svg>\n"
        "\n"
        "Example of a simple bird flying:\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" width=\"100\" height=\"100\">\n"
        "  <path d=\"M10,50 Q30,20 50,50 Q70,20 90,50 Q70,40 50,60 Q30,40 10,50 Z\" fill=\"#2c3e50\"/>\n"
        "</svg>\n"
        "\n"
        "Generate the requested subject with similar clean path coordinates and standard scale."
    )
    
    if base_svg_content:
        system_prompt += (
            "\n\nYou are provided with a high-quality base SVG template. You MUST modify this base SVG template to fulfill the user's request.\n"
            "Keep the existing layout, styles, viewBox, and structure of the base SVG where appropriate, but add, remove, or modify paths, shapes, or colors to fulfill the prompt.\n"
            f"Base SVG template:\n{base_svg_content}"
        )
        user_content = f"Modify the base SVG to show: {prompt_subject}."
    else:
        user_content = f"Generate a simple SVG of a {prompt_subject}."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        text = f"System: {system_prompt}\nUser: Generate a simple SVG of a {prompt_subject}.\nAssistant:"
    
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
    
    print("Generating SVG paths...")
    t1 = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    duration = time.time() - t1
    print(f"Generated in {duration:.2f}s")
    
    svg_content = response.strip()
    # Clean markdown if model generated it anyway
    if svg_content.startswith("```"):
        lines = svg_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        svg_content = "\n".join(lines).strip()
        
    with open(output_path, "w") as f:
        f.write(svg_content)
        
    print(f"Saved SVG to: {output_path}")
    print("\n--- SVG Code ---")
    print(svg_content)
    print("----------------")

if __name__ == "__main__":
    main()
