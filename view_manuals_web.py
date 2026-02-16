import json
import os
import gradio as gr

def load_categories():
    """Get all categories"""
    jsons_path = "./MyFixit-Dataset/jsons"
    categories = [f.replace('.json', '') for f in os.listdir(jsons_path) if f.endswith('.json')]
    return sorted(categories)

def search_guides_in_category(category, search_term=""):
    """Search guides in a category"""
    jsons_path = "./MyFixit-Dataset/jsons"
    filepath = os.path.join(jsons_path, f"{category}.json")
    
    guides = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                guide = json.loads(line.strip())
                title = guide.get('Title', '')
                if search_term.lower() in title.lower() or not search_term:
                    guides.append({'title': title, 'guide': guide})
            except:
                continue
    
    return guides

def format_guide_display(guide):
    """Format guide for display"""
    if not guide:
        return "No guide selected"
    
    output = f"# {guide.get('Title', 'Unknown Title')}\n\n"
    output += f"**Category:** {guide.get('Category', 'N/A')}\n\n"
    
    # Tools
    if 'Toolbox' in guide and guide['Toolbox']:
        output += "## Tools Needed:\n"
        for tool in guide['Toolbox']:
            output += f"- {tool.get('Name', 'Unknown tool')}\n"
        output += "\n"
    
    # Steps
    if 'Steps' in guide and guide['Steps']:
        output += "## Repair Steps:\n\n"
        for i, step in enumerate(guide['Steps'], 1):
            step_title = step.get('Title', f'Step {i}')
            output += f"### Step {i}: {step_title}\n\n"
            
            if 'Lines' in step:
                for line in step['Lines']:
                    text = line.get('Text', '')
                    if text:
                        output += f"{text}\n\n"
    
    return output

def browse_by_category(category, search_term):
    """Browse guides"""
    guides = search_guides_in_category(category, search_term)
    
    if not guides:
        return gr.Dropdown(choices=[], value=None), "No guides found"
    
    titles = [g['title'] for g in guides]
    return gr.Dropdown(choices=titles, value=titles[0] if titles else None), f"Found {len(guides)} guides"

def display_selected_guide(category, guide_title):
    """Display guide"""
    if not guide_title:
        return "Please select a guide"
    
    guides = search_guides_in_category(category, guide_title)
    
    if guides:
        return format_guide_display(guides[0]['guide'])
    else:
        return "Guide not found"

def search_by_brand(brand_name):
    """Search all categories for a brand"""
    if not brand_name:
        return "Please enter a brand name"
    
    jsons_path = "./MyFixit-Dataset/jsons"
    all_results = []
    
    for json_file in os.listdir(jsons_path):
        if not json_file.endswith('.json'):
            continue
        
        filepath = os.path.join(jsons_path, json_file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    guide = json.loads(line.strip())
                    title = guide.get('Title', '').lower()
                    category = guide.get('Category', '').lower()
                    
                    if brand_name.lower() in title or brand_name.lower() in category:
                        all_results.append({
                            'title': guide.get('Title', ''),
                            'category': guide.get('Category', ''),
                            'guide': guide
                        })
                        
                        if len(all_results) >= 20:
                            break
                except:
                    continue
        
        if len(all_results) >= 20:
            break
    
    if not all_results:
        return f"No guides found for brand: {brand_name}"
    
    output = f"# Found {len(all_results)} guides for {brand_name}\n\n"
    for result in all_results[:10]:
        output += format_guide_display(result['guide'])
        output += "\n---\n\n"
    
    if len(all_results) > 10:
        output += f"\n... and {len(all_results) - 10} more results"
    
    return output

# Interface
with gr.Blocks(title="Repair Manual Browser") as demo:
    
    gr.Markdown("# Repair Manual Browser")
    gr.Markdown("Browse and search 18,995+ repair guides")
    
    with gr.Tab("Browse by Category"):
        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=load_categories(),
                label="Select Category",
                value=load_categories()[0] if load_categories() else None
            )
            search_input = gr.Textbox(
                label="Search",
                placeholder="Enter search term..."
            )
        
        search_btn = gr.Button("Search")
        status_text = gr.Textbox(label="Status", interactive=False)
        
        guide_dropdown = gr.Dropdown(
            label="Select Guide",
            choices=[]
        )
        
        show_guide_btn = gr.Button("Show Guide Details")
        
        guide_output = gr.Markdown()
        
        search_btn.click(
            browse_by_category,
            inputs=[category_dropdown, search_input],
            outputs=[guide_dropdown, status_text]
        )
        
        show_guide_btn.click(
            display_selected_guide,
            inputs=[category_dropdown, guide_dropdown],
            outputs=guide_output
        )
    
    with gr.Tab("Search by Brand"):
        brand_input = gr.Textbox(
            label="Brand Name",
            placeholder="e.g., Samsung, iPhone, Kenmore"
        )
        brand_search_btn = gr.Button("Search")
        brand_output = gr.Markdown()
        
        brand_search_btn.click(
            search_by_brand,
            inputs=brand_input,
            outputs=brand_output
        )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, server_port=7865)