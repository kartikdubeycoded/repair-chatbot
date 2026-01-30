import sqlite3
import pandas as pd

def view_logs(limit=50):
    """View conversation logs"""
    conn = sqlite3.connect('conversation_logs.db')
    
    query = f"""
        SELECT 
            id,
            timestamp,
            user_question,
            SUBSTR(bot_response, 1, 100) as response_preview,
            sources,
            response_time_seconds
        FROM conversations
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_stats():
    """Get conversation statistics"""
    conn = sqlite3.connect('conversation_logs.db')
    cursor = conn.cursor()
    
    # Total conversations
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total = cursor.fetchone()[0]
    
    # Average response time
    cursor.execute("SELECT AVG(response_time_seconds) FROM conversations")
    avg_time = cursor.fetchone()[0] or 0
    
    conn.close()
    
    print(f" Conversation Statistics")
    print(f"{'='*50}")
    print(f"Total conversations: {total}")
    print(f"Average response time: {avg_time:.2f} seconds")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    get_stats()
    print("Recent conversations:")
    print(view_logs(10))