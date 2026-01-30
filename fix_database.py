import sqlite3
import os

print("Fixing database structure...\n")

# Check if database exists
if os.path.exists('conversation_logs.db'):
    print("Found existing database")
    
    conn = sqlite3.connect('conversation_logs.db')
    cursor = conn.cursor()
    
    # Check if context_info column exists
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [column[1] for column in cursor.fetchall()]
    
    print(f"Current columns: {columns}")
    
    if 'context_info' not in columns:
        print("\nAdding context_info column...")
        cursor.execute("ALTER TABLE conversations ADD COLUMN context_info TEXT")
        conn.commit()
        print("✓ Column added!")
    else:
        print("\n✓ context_info column already exists!")
    
    conn.close()
else:
    print("No existing database found - will be created fresh")

print("\n✓ Database ready!")