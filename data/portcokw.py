"""
Read mercury.csv, enrich with OpenAI, output enriched CSV
"""

import csv
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_company(company_name: str, description: str) -> dict:
    """
    Use OpenAI to extract keywords, target customer, and competitors
    """
    prompt = f"""Analyze this company and extract structured information.

Company: {company_name}
Description: {description}

Please provide:
1. Keywords: 5-10 relevant keywords (technology, industry, business model). Separate with commas.
2. Target Customer: Who are their primary customers? Be specific (e.g., "dental practices", "mid-market manufacturers")
3. What They Build: A concise 5-10 word summary of their core product/service
4. Competitors: 3-5 likely competitor company names. Separate with commas. If not obvious, leave empty.

Return ONLY valid JSON in this exact format:
{{
  "keywords": "keyword1,keyword2,keyword3",
  "target_customer": "specific target customer description",
  "what_they_build": "concise product/service summary",
  "competitors": "competitor1,competitor2,competitor3"
}}

Be specific and actionable."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing B2B companies. Always return valid JSON with comma-separated strings."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "keywords": "",
            "target_customer": "",
            "what_they_build": "",
            "competitors": ""
        }


def enrich_csv(input_file='mercury.csv', output_file='mercury_enriched.csv'):
    """
    Read mercury.csv, enrich with AI, output new CSV
    """
    enriched_rows = []
    
    # Read input CSV
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} companies\n")
    
    # Process each company
    for idx, row in enumerate(rows, 1):
        company_name = row['company_name']
        description = row['description']
        
        print(f"[{idx}/{len(rows)}] Processing: {company_name}")
        
        # Get AI analysis
        analysis = analyze_company(company_name, description)
        
        # Merge with existing data
        enriched_row = {
            'company_name': row['company_name'],
            'website': row['website'],
            'description': row['description'],
            'industry': row['industry'],
            'headquarters': row['headquarters'],
            'keywords': analysis.get('keywords', ''),
            'target_customer': analysis.get('target_customer', ''),
            'what_they_build': analysis.get('what_they_build', ''),
            'competitors': analysis.get('competitors', '')
        }
        
        enriched_rows.append(enriched_row)
        
        print(f"  Keywords: {analysis.get('keywords', 'N/A')[:60]}...")
        print(f"  Target: {analysis.get('target_customer', 'N/A')[:60]}...")
        print()
        
        # Rate limiting (to avoid hitting OpenAI limits)
        time.sleep(0.5)
    
    # Write output CSV
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'company_name',
            'website', 
            'description',
            'industry',
            'headquarters',
            'keywords',
            'target_customer',
            'what_they_build',
            'competitors'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    
    print(f"Done! Enriched data saved to {output_file}")
    print(f"\nNow you can:")
    print(f"1. Open Supabase Table Editor")
    print(f"2. Go to portfolio_companies table")
    print(f"3. Click 'Insert' → 'Import data from spreadsheet'")
    print(f"4. Upload {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enrich mercury.csv with AI-generated data')
    parser.add_argument('--input', default='Quicksilver\mercury.csv', help='Input CSV file')
    parser.add_argument('--output', default='mercury_enriched.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    enrich_csv(args.input, args.output)