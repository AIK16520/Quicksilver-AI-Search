# config.py

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# ===== Environment Variables =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional for now
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# ===== Validation =====
# Check required variables (Supabase only for now)
required_vars = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
}

missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# OpenAI is optional for now
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Processor will not work.")

# ===== Initialize Supabase Client =====
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== Constants =====
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
GPT_MODEL = "gpt-4o-mini"
MAX_TOKENS_GPT = 4000
TEMPERATURE = 0.3