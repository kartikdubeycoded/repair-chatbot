import json
import os

def list_categories():
    """List all available categories"""
    jsons_path = "./MyFixit-Dataset/jsons"
    json_files = [f.replace('.json', '') for f in os.listdir(jsons_path) if f.endswith('.json')]
    
    print("\nAvailable Categories:")
    print("="*60)
    for i, category in enumerate(json_files, 1):
        print(f"{i}. {category}")
    print("="*60)
    
    return json_files

def search_guides(category, search_term=""):
    """Search for repair guides in a category"""
    jsons_path = "./MyFixit-Dataset/jsons"
    filepath = os.path.join(jsons_path, f"{category}.json")
    
    guides = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                guide = json.loads(line.strip())
                if search_term.lower() in guide.get('Title', '').lower():
                    guides.append(guide)
            except:
                continue
    
    return guides

def display_guide(guide):
    """Display a repair guide in readable format"""
    print("\n" + "="*80)
    print(f"REPAIR GUIDE")
    print("="*80)
    
    print(f"\nTitle: {guide.get('Title', 'N/A')}")
    print(f"Category: {guide.get('Category', 'N/A')}")
    print(f"Guide ID: {guide.get('Guidid', 'N/A')}")
    
    # Display tools needed
    if 'Toolbox' in guide and guide['Toolbox']:
        print("\nTools Needed:")
        print("-"*80)
        for tool in guide['Toolbox']:
            print(f"  - {tool.get('Name', 'Unknown tool')}")
    
    # Display repair steps
    if 'Steps' in guide and guide['Steps']:
        print("\nRepair Steps:")
        print("-"*80)
        for i, step in enumerate(guide['Steps'], 1):
            step_title = step.get('Title', f'Step {i}')
            print(f"\nStep {i}: {step_title}")
            
            # Get step instructions
            if 'Lines' in step:
                for line in step['Lines']:
                    text = line.get('Text', '')
                    if text:
                        print(f"  {text}")
    
    print("\n" + "="*80)

def browse_by_brand(brand_name):
    """Find all guides for a specific brand"""
    jsons_path = "./MyFixit-Dataset/jsons"
    all_guides = []
    
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
                        all_guides.append({
                            'title': guide.get('Title'),
                            'category': guide.get('Category'),
                            'guide': guide
                        })
                except:
                    continue
    
    return all_guides

def interactive_viewer():
    """Interactive manual viewer"""
    print("\n" + "="*80)
    print("REPAIR MANUAL VIEWER")
    print("="*80)
    
    while True:
        print("\nOptions:")
        print("1. Browse by category")
        print("2. Search by brand")
        print("3. Search by keyword")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            categories = list_categories()
            cat_num = input("\nEnter category number: ").strip()
            
            try:
                cat_idx = int(cat_num) - 1
                category = categories[cat_idx]
                
                search_term = input(f"Search in {category} (or press Enter for all): ").strip()
                guides = search_guides(category, search_term)
                
                print(f"\nFound {len(guides)} guides:")
                for i, guide in enumerate(guides[:20], 1):
                    print(f"{i}. {guide.get('Title')}")
                
                if guides:
                    guide_num = input("\nEnter guide number to view (or Enter to skip): ").strip()
                    if guide_num:
                        guide_idx = int(guide_num) - 1
                        if 0 <= guide_idx < len(guides):
                            display_guide(guides[guide_idx])
            except:
                print("Invalid selection")
        
        elif choice == '2':
            brand = input("\nEnter brand name (e.g., Kenmore, Samsung, iPhone): ").strip()
            guides = browse_by_brand(brand)
            
            print(f"\nFound {len(guides)} guides for {brand}:")
            for i, item in enumerate(guides[:20], 1):
                print(f"{i}. {item['title']} ({item['category']})")
            
            if guides:
                guide_num = input("\nEnter guide number to view (or Enter to skip): ").strip()
                if guide_num:
                    guide_idx = int(guide_num) - 1
                    if 0 <= guide_idx < len(guides):
                        display_guide(guides[guide_idx]['guide'])
        
        elif choice == '3':
            keyword = input("\nEnter search keyword: ").strip()
            all_guides = []
            
            jsons_path = "./MyFixit-Dataset/jsons"
            for json_file in os.listdir(jsons_path):
                if not json_file.endswith('.json'):
                    continue
                    
                filepath = os.path.join(jsons_path, json_file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            guide = json.loads(line.strip())
                            if keyword.lower() in json.dumps(guide).lower():
                                all_guides.append(guide)
                                if len(all_guides) >= 20:
                                    break
                        except:
                            continue
                
                if len(all_guides) >= 20:
                    break
            
            print(f"\nFound {len(all_guides)} guides:")
            for i, guide in enumerate(all_guides, 1):
                print(f"{i}. {guide.get('Title')}")
            
            if all_guides:
                guide_num = input("\nEnter guide number to view (or Enter to skip): ").strip()
                if guide_num:
                    guide_idx = int(guide_num) - 1
                    if 0 <= guide_idx < len(all_guides):
                        display_guide(all_guides[guide_idx])
        
        elif choice == '4':
            print("\nExiting manual viewer...")
            break
        else:
            print("\nInvalid choice")

if __name__ == "__main__":
    interactive_viewer()