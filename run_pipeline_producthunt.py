#!/usr/bin/env python3
"""
Simple runner for Product Hunt Daily Pipeline

This is a convenience wrapper that you can call from anywhere.
Fetches today's top products from Product Hunt and stores in Supabase.

Usage:
    python run_pipeline_producthunt.py           # Get today's top 50
    python run_pipeline_producthunt.py --limit 100  # Get top 100
    python run_pipeline_producthunt.py --test    # Test without storing
"""

import sys
import subprocess
from pathlib import Path

# Get the path to the actual pipeline script
pipeline_script = Path(__file__).parent / "pipeline" / "run_producthunt_daily.py"

# Forward all arguments to the pipeline script
result = subprocess.run([sys.executable, str(pipeline_script)] + sys.argv[1:])
sys.exit(result.returncode)

