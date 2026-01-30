# Repair Chatbot - Local AI Assistant

A context-aware AI chatbot that helps service engineers find repair information from 18,995+ repair manuals. Runs completely offline using a local LLM.

## Features

- Context-aware conversations (remembers brand/model across questions)
- Smart clarifying questions when information is vague
- Real-time streaming responses
- 18,995 repair guides across multiple categories
- Conversation logging with Excel export
- Works completely offline after initial setup
- No API keys required

## System Requirements

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but recommended)
  - CPU-only mode supported but slower
- **Storage**: ~15GB free space
  - Model: ~2GB
  - Vector database: ~500MB
  - Dataset: ~200MB

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/repair-chatbot.git
cd repair-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch (GPU version for NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

For CPU-only:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 4. Download the Dataset

```bash
git clone https://github.com/rub-ksv/MyFixit-Dataset.git
```

### 5. Download the AI Model

Download the Qwen 2.5 3B model:

**Option A: Direct Download**
1. Go to https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
2. Download `qwen2.5-3b-instruct-q4_k_m.gguf`
3. Create `models/` folder and place the file there

**Option B: Using huggingface-cli**
```bash
pip install huggingface_hub
mkdir models
cd models
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir .
cd ..
```

### 6. Build the Vector Database

This processes all repair manuals and creates a searchable database:

```bash
python build_vector_db.py
```

This takes 2-3 minutes and creates a `chroma_db/` folder.

### 7. Download Embedding Model (For Offline Use)

**IMPORTANT: Run this WITH internet connection**

```bash
python download_embeddings_for_offline.py
```

This downloads and caches a small model (~80MB). After this, the chatbot works completely offline.

## Usage

### Start the Chatbot

```bash
python chatbot_smart.py
```

The chatbot will:
1. Load the AI model
2. Start a web interface
3. Automatically open in your browser

### Example Conversations

**Vague question (bot asks for clarification):**
```
You: My washing machine is broken
Bot: I can help! To find the right repair guide, I need:
     1. What brand/model is your washing machine?
     2. What exactly is the problem?

You: Kenmore Elite HE3, won't drain
Bot: [Searches Kenmore Elite HE3 guides and provides specific answer]

You: What tools do I need?
Bot: [Remembers we're discussing Kenmore Elite HE3, provides tools from same manual]
```

**Specific question (bot answers directly):**
```
You: My Samsung RF28R7201SR refrigerator is not cooling
Bot: [Provides specific repair steps for this model]
```

## Additional Tools

### View Conversation Logs

```bash
python view_logs.py
```

### Export Logs to Excel

```bash
python export_logs_to_excel.py
```

Creates a formatted Excel file with statistics.

### Browse Repair Manuals

```bash
python view_manuals.py
```

Interactive tool to browse and search the repair database.

## Project Structure

```
repair-chatbot/
├── chatbot_smart.py              # Main chatbot application
├── context_manager.py            # Context tracking and clarification logic
├── smart_search.py              # Vector search and filtering
├── build_vector_db.py           # Creates searchable database
├── view_logs.py                 # View conversation history
├── export_logs_to_excel.py      # Export logs to Excel
├── view_manuals.py              # Browse repair manuals
├── download_embeddings_for_offline.py  # Setup for offline use
├── fix_database.py              # Database maintenance
├── requirements.txt             # Python dependencies
├── models/                      # AI model folder
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
├── MyFixit-Dataset/            # Repair manuals dataset
├── chroma_db/                  # Vector database (generated)
└── conversation_logs.db        # SQLite conversation history (generated)
```

## Configuration

### Using CPU Instead of GPU

Edit `chatbot_smart.py` and `smart_search.py`:

Change:
```python
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

To:
```python
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

Also in `chatbot_smart.py`:
```python
llm = Llama(
    model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=0,  # Change from -1 to 0 for CPU-only
    n_batch=512,
    n_threads=8,
    verbose=False
)
```

## Troubleshooting

### "Cannot send a request, as the client has been closed"

You're trying to run offline but the embedding model isn't cached yet.

**Solution**: Connect to internet and run:
```bash
python download_embeddings_for_offline.py
```

### "Module not found" errors

Make sure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Slow responses (30+ seconds)

- Check if GPU is being used: Look for "Loading model to GPU..." in terminal
- Reduce max_tokens in `chatbot_smart.py` (line ~145) from 300 to 150
- Use CPU with more threads: Change `n_threads=8` to `n_threads=16`

### "CUDA out of memory"

Your GPU doesn't have enough VRAM. Switch to CPU mode (see Configuration section above).

### Database errors

Reset the database:
```bash
python fix_database.py
```

## Performance Tips

- **GPU highly recommended** for good performance (5-10 second responses)
- **CPU mode** will work but responses take 20-30 seconds
- Reduce `max_tokens` for faster responses
- Close other GPU-intensive applications

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project uses the MyFixit Dataset which is subject to its own license.
See: https://github.com/rub-ksv/MyFixit-Dataset

## Acknowledgments

- Dataset: MyFixit-Dataset by rub-ksv
- Model: Qwen 2.5 3B by Alibaba Cloud
- Embeddings: all-MiniLM-L6-v2 by sentence-transformers

## Support

For issues or questions, please open an issue on GitHub.

---

**Note**: This chatbot is for educational and assistance purposes. Always verify repair information with official manufacturer documentation before performing repairs.
