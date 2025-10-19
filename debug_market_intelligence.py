#!/usr/bin/env python3
"""
Debug script to check market intelligence response structure
"""

import requests
import json

def debug_market_intelligence():
    """Debug the market intelligence response structure"""
    print("🔍 Debugging Market Intelligence Response Structure...")
    
    try:
        response = requests.post('http://localhost:8000/market-intelligence', 
            json={
                'query': 'AI productivity tools',
                'max_results_per_dimension': 3,
                'include_ai_insights': False,
                'format_type': 'api'
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Market Intelligence API Response: SUCCESS")
            
            dimensions = data['data'].get('dimensions', {})
            print(f"📊 Dimensions found: {list(dimensions.keys())}")
            
            total_ph_products = 0
            for dim_name, dim_data in dimensions.items():
                if isinstance(dim_data, dict) and 'articles' in dim_data:
                    articles = dim_data['articles']
                    print(f"📰 {dim_name}: {len(articles)} articles")
                    
                    # Check for Product Hunt products
                    ph_articles = [a for a in articles if a.get('source') == 'product_hunt']
                    if ph_articles:
                        print(f"  🚀 Product Hunt products: {len(ph_articles)}")
                        for ph in ph_articles[:2]:
                            title = ph.get('title', 'Unknown')
                            print(f"    - {title}")
                        total_ph_products += len(ph_articles)
                    else:
                        print(f"  ❌ No Product Hunt products in {dim_name}")
            
            print(f"🚀 Total Product Hunt products: {total_ph_products}")
            
            if total_ph_products == 0:
                print("\n🔍 Checking if Product Hunt search is being called...")
                # Check if the market intelligence service is actually calling Product Hunt search
                print("This suggests the market intelligence service may not be calling Product Hunt search methods")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    debug_market_intelligence()

