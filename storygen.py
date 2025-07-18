import argparse
import json
import os
import openai
from tqdm import tqdm


def estimate_tokens(text):
    words = len(text.split())
    return int(words * 1.5)


def get_last_n_words(text, n):
    words = text.split()
    return " ".join(words[-n:])


def generate_text(prompt, api_key, model="gpt-4o", max_tokens=1500):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional writer. Follow instructions carefully, continue logically."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def collect_options(node, inherited_options):
    return {**inherited_options, **(node.get('options') or {})}


def gather_previous_content(nodes):
    content = []

    def recurse(n):
        if n.get('content'):
            content.append(n['content'])
        for child in n.get('nodes', []):
            recurse(child)

    for node in nodes:
        recurse(node)

    return "\n\n".join(content)


def process_ai_node(node, context, options, api_key, model, token_budget):
    genre = options.get('genre', 'General Fiction')
    tone = options.get('tone', 'Neutral')
    style = options.get('style', 'Standard prose')
    audience = options.get('audience', 'General')
    output_format = options.get('output_format', 'Narrative text')

    summary = node.get('summary', '')
    accumulated = ""
    current_prompt = f"""
Write the beginning of this scene.
Genre: {genre}
Tone: {tone}
Style: {style}
Audience: {audience}
Format: {output_format}
Summary: {summary}
Do not repeat previous content. Previous content:
{context}
"""

    with tqdm(total=token_budget, desc=f"Generating {node['name']}") as pbar:
        while estimate_tokens(accumulated) < token_budget:
            result = generate_text(current_prompt, api_key=api_key, model=model, max_tokens=1500)
            accumulated += "\n\n" + result
            last_chunk = get_last_n_words(accumulated, 150)
            current_prompt = f"""
Continue smoothly from here without repeating or contradicting:
{last_chunk}
Keep tone, style, and logical continuity.
"""
            pbar.update(min(estimate_tokens(result), token_budget - pbar.n))

    node['content'] = accumulated.strip()


def walk_nodes(node, context_chain, inherited_options, api_key, model, token_budget):
    if node['type'] == 'ai_generated_text':
        previous_content = gather_previous_content(context_chain)
        process_ai_node(node, previous_content, inherited_options, api_key, model, token_budget)
        context_chain.append(node)
    elif 'nodes' in node:
        context_chain.append(node)
        for child in node['nodes']:
            walk_nodes(child, context_chain, collect_options(child, inherited_options), api_key, model, token_budget)
        context_chain.pop()


def output_text_file(book_data, output_path):
    lines = []

    def recurse(node, depth=0):
        indent = " " * (depth * 4)
        if node.get('name'):
            lines.append(f"{indent}{node['name']}\n")
        if node.get('content'):
            lines.append(f"{indent}{node['content']}\n")
        for child in node.get('nodes', []):
            recurse(child, depth + 1)

    recurse(book_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate AI-written book/article from JSON structure.")
    parser.add_argument("--source", required=True, help="Path to JSON structure file.")
    parser.add_argument("--api-key", required=True, help="OpenAI API key.")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    parser.add_argument("--token-budget", type=int, default=3000, help="Token budget per ai_generated_text node.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument("--output-format", choices=["txt", "json"], default="txt", help="Output file format (txt/json).")

    args = parser.parse_args()

    with open(args.source, 'r', encoding='utf-8') as f:
        book_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    walk_nodes(
        book_data,
        context_chain=[],
        inherited_options=book_data.get('options', {}),
        api_key=args.api_key,
        model=args.model,
        token_budget=args.token_budget
    )

    base_name = os.path.splitext(os.path.basename(args.source))[0]
    if args.output_format == "txt":
        output_path = os.path.join(args.output_dir, f"{base_name}.txt")
        output_text_file(book_data, output_path)
    else:
        output_path = os.path.join(args.output_dir, f"{base_name}_with_content.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(book_data, f, indent=4, ensure_ascii=False)

    print(f"\nâ Done. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
