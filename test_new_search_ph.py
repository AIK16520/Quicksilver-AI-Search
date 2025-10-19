#!/usr/bin/env python3
"""
Test script to check if Product Hunt products appear in NEW SEARCH (Market Intelligence)
"""

import requests
import json

def test_new_search_producthunt():
    """Test NEW SEARCH with Product Hunt integration"""
    print("🔍 Testing NEW SEARCH (Market Intelligence) with Product Hunt integration...")
    print("=" * 60)
    
    try:
        response = requests.post('http://localhost:8000/market-intelligence', 
            json={
                'query': 'AI productivity tools',
                'max_results_per_dimension': 5,
                'include_ai_insights': True,
                'format_type': 'api'
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Market Intelligence API Response: SUCCESS")
            
            # Check for dimensional results
            if 'data' in data and 'dimensional_results' in data['data']:
                dimensional_results = data['data']['dimensional_results']
                print(f"📊 Found {len(dimensional_results)} dimensions")
                
                # Check for Product Hunt products in each dimension
                total_ph_products = 0
                for dimension_name, dimension_data in dimensional_results.items():
                    if 'articles' in dimension_data:
                        ph_products = [a for a in dimension_data['articles'] if a.get('source') == 'product_hunt']
                        if ph_products:
                            print(f"🚀 {dimension_name}: {len(ph_products)} Product Hunt products")
                            for product in ph_products[:2]:
                                product_name = product.get('title', 'Unknown')
                                weight_boost = product.get('weight_boost', 1.0)
                                print(f"  - {product_name} (boost: {weight_boost}x)")
                            total_ph_products += len(ph_products)
                
                print(f"🚀 Total Product Hunt products found: {total_ph_products}")
                
                if total_ph_products == 0:
                    print("❌ No Product Hunt products found in market intelligence results")
            else:
                print("❌ No dimensional_results found in response")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("💡 Make sure the API server is running on localhost:8000")

if __name__ == "__main__":
    test_new_search_producthunt()

