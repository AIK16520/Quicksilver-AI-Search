# build_portfolio_maps.py
"""
Build comprehensive mapping files from portfolio companies for intelligent search.

This script creates:
1. keyword_to_companies.json - Maps each keyword to list of companies
2. industry_to_companies.json - Maps each industry to list of companies
3. all_keywords.json - List of all unique keywords
4. all_industries.json - List of all unique industries
"""

import json
from collections import defaultdict
from config import supabase_client

def build_portfolio_maps():
    """Fetch portfolio companies and build mapping structures."""

    print("Fetching portfolio companies from database...")

    # Fetch all portfolio companies
    response = supabase_client.table('portfolio_companies').select(
        'company_name, industry, keywords, what_they_build'
    ).execute()

    companies = response.data or []
    print(f"Loaded {len(companies)} portfolio companies")

    # Initialize mapping structures
    keyword_to_companies = defaultdict(list)
    industry_to_companies = defaultdict(list)
    all_keywords = set()
    all_industries = set()

    # Process each company
    for company in companies:
        company_name = company.get('company_name', '')
        industry = company.get('industry', '')
        keywords = company.get('keywords', [])
        what_they_build = company.get('what_they_build', '')

        # Map industry to company
        if industry:
            industry_clean = industry.strip()
            industry_to_companies[industry_clean].append(company_name)
            all_industries.add(industry_clean)

        # Map keywords to company
        if keywords:
            for keyword in keywords:
                if keyword:
                    keyword_clean = keyword.strip()
                    keyword_to_companies[keyword_clean].append(company_name)
                    all_keywords.add(keyword_clean)

        # Extract keywords from what_they_build (common tech terms)
        # This helps with matching queries like "AI" or "blockchain"
        if what_they_build:
            what_they_build_lower = what_they_build.lower()
            # Add industry as keyword too
            if industry:
                keyword_to_companies[industry.strip()].append(company_name)
                all_keywords.add(industry.strip())

    # Convert defaultdict to regular dict for JSON serialization
    keyword_to_companies = dict(keyword_to_companies)
    industry_to_companies = dict(industry_to_companies)

    # Convert sets to sorted lists
    all_keywords = sorted(list(all_keywords))
    all_industries = sorted(list(all_industries))

    # Save to JSON files
    print("\nSaving mapping files...")

    with open('keyword_to_companies.json', 'w', encoding='utf-8') as f:
        json.dump(keyword_to_companies, f, indent=2, ensure_ascii=False)
    print(f"[OK] keyword_to_companies.json - {len(keyword_to_companies)} keywords")

    with open('industry_to_companies.json', 'w', encoding='utf-8') as f:
        json.dump(industry_to_companies, f, indent=2, ensure_ascii=False)
    print(f"[OK] industry_to_companies.json - {len(industry_to_companies)} industries")

    with open('all_keywords.json', 'w', encoding='utf-8') as f:
        json.dump(all_keywords, f, indent=2, ensure_ascii=False)
    print(f"[OK] all_keywords.json - {len(all_keywords)} unique keywords")

    with open('all_industries.json', 'w', encoding='utf-8') as f:
        json.dump(all_industries, f, indent=2, ensure_ascii=False)
    print(f"[OK] all_industries.json - {len(all_industries)} unique industries")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Portfolio Companies: {len(companies)}")
    print(f"Unique Keywords: {len(all_keywords)}")
    print(f"Unique Industries: {len(all_industries)}")
    print(f"\nTop 10 Keywords by Company Count:")
    keyword_counts = [(kw, len(companies)) for kw, companies in keyword_to_companies.items()]
    keyword_counts.sort(key=lambda x: x[1], reverse=True)
    for kw, count in keyword_counts[:10]:
        print(f"  - {kw}: {count} companies")

    print(f"\nTop 10 Industries by Company Count:")
    industry_counts = [(ind, len(companies)) for ind, companies in industry_to_companies.items()]
    industry_counts.sort(key=lambda x: x[1], reverse=True)
    for ind, count in industry_counts[:10]:
        print(f"  - {ind}: {count} companies")

    print("\n" + "="*60)
    print("Mapping files created successfully!")
    print("="*60)

    return {
        'keyword_to_companies': keyword_to_companies,
        'industry_to_companies': industry_to_companies,
        'all_keywords': all_keywords,
        'all_industries': all_industries
    }


if __name__ == "__main__":
    build_portfolio_maps()
