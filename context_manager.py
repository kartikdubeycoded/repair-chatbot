def get_conversation_history(history):
    """Extract clean conversation history from Gradio format"""
    
    conversation = []
    
    for message in history:
        role = message['role']
        text = message['content'][0]['text']
        
        conversation.append({
            'role': role,
            'text': text
        })
    
    return conversation

def get_last_user_messages(history, count=3):
    """Get the last N user messages"""
    
    user_messages = []
    
    for message in history:
        if message['role'] == 'user':
            user_messages.append(message['content'][0]['text'])
    
    # Return last N messages
    return user_messages[-count:] if user_messages else []

def build_context_string(history):
    """Build a context string from conversation history"""
    
    context_parts = []
    
    for message in history:
        role = message['role']
        text = message['content'][0]['text']
        
        if role == 'user':
            context_parts.append(f"User asked: {text}")
        else:
            # Only include short version of assistant response
            short_text = text[:100] + "..." if len(text) > 100 else text
            context_parts.append(f"Assistant said: {short_text}")
    
    return "\n".join(context_parts)

# Test it
if __name__ == "__main__":
    # Sample history
    test_history = [
        {'role': 'user', 'content': [{'text': 'my washing machine is broken'}]},
        {'role': 'assistant', 'content': [{'text': 'Can you tell me the brand?'}]},
        {'role': 'user', 'content': [{'text': 'Kenmore Elite HE3'}]},
    ]
    
    print("Conversation history:")
    print(get_conversation_history(test_history))
    
    print("\nLast 2 user messages:")
    print(get_last_user_messages(test_history, 2))
    
    print("\nContext string:")
    print(build_context_string(test_history))

def is_vague_question(message, history):
    """
    Detect if the question is too vague and needs clarification.
    
    Returns: (is_vague, reason)
    """
    
    message_lower = message.lower()
    
    # Check if it's the first message
    is_first_message = len(history) == 0
    
    # Vague keywords that indicate lack of specificity
    vague_phrases = [
        'not working', 'broken', 'problem', 'issue', 'won\'t work',
        'doesn\'t work', 'stopped working', 'having trouble', 'not cooling',
        'won\'t start', 'won\'t drain', 'leaking', 'making noise'
    ]
    
    # Specific appliance types (without brand/model)
    appliance_types = [
        'washing machine', 'washer', 'dryer', 'dishwasher', 
        'refrigerator', 'fridge', 'oven', 'microwave',
        'phone', 'laptop', 'computer', 'tablet', 'camera',
        'tv', 'television', 'monitor'
    ]
    
    # Common brands
    brands = [
        'kenmore', 'whirlpool', 'samsung', 'lg', 'ge', 'maytag',
        'bosch', 'frigidaire', 'electrolux', 'kitchenaid',
        'apple', 'dell', 'hp', 'lenovo', 'asus', 'sony', 'canon', 'nikon',
        'iphone', 'macbook', 'ipad'
    ]
    
    # Check conditions
    has_vague_phrase = any(phrase in message_lower for phrase in vague_phrases)
    has_appliance = any(appliance in message_lower for appliance in appliance_types)
    has_brand = any(brand in message_lower for brand in brands)
    
    # Model number pattern (e.g., "HE3", "RF28R7201SR", "SHU5315UC")
    import re
    has_model_pattern = bool(re.search(r'\b[A-Z]{2,}\d+[A-Z]*\b|\b\d+[A-Z]+\d*\b', message))
    
    # Decision logic - only for FIRST message
    if is_first_message:
        # Has appliance but no brand/model = VAGUE
        if has_appliance and not has_brand and not has_model_pattern:
            return True, "missing_brand"
        
        # Has vague phrase without specifics = VAGUE  
        if has_vague_phrase and not has_brand and not has_model_pattern:
            return True, "missing_specifics"
    
    # Follow-up questions are not vague (we have context)
    return False, "specific_enough"

def needs_model_clarification(message, history):
    """Check if we need to ask for model information"""
    
    is_vague, reason = is_vague_question(message, history)
    
    if is_vague:
        return True, reason
    
    return False, None

# Test it
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Vague Question Detection")
    print("="*60)
    
    test_cases = [
        ("my washing machine is broken", []),
        ("My Kenmore Elite HE3 won't drain", []),
        ("Bosch dishwasher SHU5315UC leaking water", []),
        ("refrigerator not cooling", []),
        ("what tools do i need", [{'role': 'user', 'content': [{'text': 'kenmore washer'}]}]),
    ]
    
    for message, history in test_cases:
        is_vague, reason = is_vague_question(message, history)
        print(f"\nMessage: '{message}'")
        print(f"Is vague: {is_vague}")
        print(f"Reason: {reason}")

def generate_clarification_prompt(message, reason):
    """Generate a clarifying question based on the vague message"""
    
    message_lower = message.lower()
    
    # Detect appliance type
    appliance_type = "appliance"
    if 'washing machine' in message_lower or 'washer' in message_lower:
        appliance_type = "washing machine"
    elif 'dryer' in message_lower:
        appliance_type = "dryer"
    elif 'dishwasher' in message_lower:
        appliance_type = "dishwasher"
    elif 'refrigerator' in message_lower or 'fridge' in message_lower:
        appliance_type = "refrigerator"
    elif 'phone' in message_lower:
        appliance_type = "phone"
    elif 'laptop' in message_lower or 'computer' in message_lower:
        appliance_type = "computer"
    
    prompt = f"""I can help you with your {appliance_type} issue! To find the right repair guide, I need a bit more information:

1. **What brand/model** is your {appliance_type}?
   (Example: Kenmore Elite HE3, Bosch SHU5315UC, Samsung RF28R7201SR)

2. **What exactly is the problem?**
   (Example: won't drain, making loud noise, not cooling, won't start)

Please provide these details so I can find the specific repair guide for you! 🔧"""
    
    return prompt

# Update the test section
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Clarification Prompts")
    print("="*60)
    
    vague_messages = [
        "my washing machine is broken",
        "refrigerator not cooling",
        "phone screen cracked"
    ]
    
    for msg in vague_messages:
        is_vague, reason = is_vague_question(msg, [])
        if is_vague:
            print(f"\nVague message: '{msg}'")
            print(generate_clarification_prompt(msg, reason))
            print("-" * 60)


def extract_brand_and_model(message):
    """
    Extract brand and model from user message.
    
    Returns: {
        'brand': 'Kenmore',
        'model': 'Elite HE3',
        'full_text': 'Kenmore Elite HE3'
    }
    """
    
    import re
    
    message_lower = message.lower()
    
    # Common brands (case-insensitive)
    brands = {
        'kenmore': 'Kenmore',
        'whirlpool': 'Whirlpool',
        'samsung': 'Samsung',
        'lg': 'LG',
        'ge': 'GE',
        'maytag': 'Maytag',
        'bosch': 'Bosch',
        'frigidaire': 'Frigidaire',
        'electrolux': 'Electrolux',
        'kitchenaid': 'KitchenAid',
        'apple': 'Apple',
        'dell': 'Dell',
        'hp': 'HP',
        'lenovo': 'Lenovo',
        'asus': 'ASUS',
        'sony': 'Sony',
        'canon': 'Canon',
        'nikon': 'Nikon',
        'iphone': 'iPhone',
        'macbook': 'MacBook',
        'ipad': 'iPad'
    }
    
    # Find brand
    found_brand = None
    for brand_key, brand_name in brands.items():
        if brand_key in message_lower:
            found_brand = brand_name
            break
    
    # Find model number pattern
    # Matches: HE3, RF28R7201SR, SHU5315UC, Elite HE3, etc.
    model_patterns = [
        r'\b([A-Z]{2,}[-\s]?\d+[A-Z\d\-]*)\b',  # HE3, SHU5315UC
        r'\b(Elite\s+[A-Z0-9]+)\b',              # Elite HE3
        r'\b(\d+[A-Z]+\d+[A-Z\d]*)\b',          # RF28R7201SR
    ]
    
    found_model = None
    for pattern in model_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            found_model = match.group(1)
            break
    
    # Build full text
    full_text = None
    if found_brand and found_model:
        full_text = f"{found_brand} {found_model}"
    elif found_brand:
        full_text = found_brand
    elif found_model:
        full_text = found_model
    
    return {
        'brand': found_brand,
        'model': found_model,
        'full_text': full_text
    }

# Update test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Brand/Model Extraction")
    print("="*60)
    
    test_responses = [
        "Kenmore Elite HE3",
        "It's a Samsung RF28R7201SR",
        "Bosch dishwasher model SHU5315UC",
        "LG washer",
        "iPhone 12 Pro",
        "just a whirlpool washer",
    ]
    
    for response in test_responses:
        result = extract_brand_and_model(response)
        print(f"\nInput: '{response}'")
        print(f"Brand: {result['brand']}")
        print(f"Model: {result['model']}")
        print(f"Full: {result['full_text']}")