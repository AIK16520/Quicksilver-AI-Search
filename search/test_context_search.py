"""
Test script for context-aware search system
Tests intent detection and context building without making API calls
"""

from search import SearchService

def test_intent_detection():
    """Test that different query types are detected correctly"""

    searcher = SearchService()

    test_cases = [
        # Portfolio mode tests
        ("any news on our portfolio companies", "portfolio_mode"),
        ("updates about our investments", "portfolio_mode"),
        ("portfolio company updates", "portfolio_mode"),

        # Industry mode tests
        ("what's happening in AI infrastructure industry", "industry_mode"),
        ("fintech sector trends", "industry_mode"),
        ("vertical SaaS market landscape", "industry_mode"),

        # Company mode tests
        ("news about OpenAI", "company_mode"),
        ("Stripe latest developments", "company_mode"),
        ("random company search", "company_mode"),
    ]

    print("Testing Intent Detection\n" + "="*60)

    for query, expected_mode in test_cases:
        mode, entity = searcher._detect_query_type(query)
        status = "✓" if mode == expected_mode else "✗"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_mode}, Got: {mode}")
        if entity:
            print(f"   Entity: {entity}")
        print()

def test_context_building():
    """Test that context is built correctly for each mode"""

    searcher = SearchService()

    print("\nTesting Context Building\n" + "="*60)

    # Test portfolio mode
    print("\n1. Portfolio Mode:")
    mode, entity = searcher._detect_query_type("portfolio updates")
    context = searcher._build_search_context(mode, entity)
    print(f"   Mode: {context['mode']}")
    print(f"   Description: {context['description']}")
    print(f"   Entities: {len(context['entities'])} companies")
    if context['entities']:
        print(f"   First 3: {context['entities'][:3]}")

    # Test industry mode
    print("\n2. Industry Mode:")
    mode, entity = searcher._detect_query_type("AI infrastructure industry")
    context = searcher._build_search_context(mode, entity)
    print(f"   Mode: {context['mode']}")
    print(f"   Description: {context['description']}")
    print(f"   Entities: {len(context['entities'])} total")
    if context['entities']:
        print(f"   Sample: {context['entities'][:3]}")

    # Test company mode
    print("\n3. Company Mode:")
    companies = searcher._get_portfolio_companies()
    if companies:
        test_company = companies[0]['company_name']
        mode, entity = searcher._detect_query_type(f"news about {test_company}")
        context = searcher._build_search_context(mode, entity)
        print(f"   Mode: {context['mode']}")
        print(f"   Description: {context['description']}")
        print(f"   Entities: {len(context['entities'])} (company + competitors)")
        if context['entities']:
            print(f"   Companies: {context['entities']}")

def test_result_filtering():
    """Test result filtering and scoring"""

    from search import SearchService

    searcher = SearchService()

    print("\n\nTesting Result Filtering\n" + "="*60)

    # Mock results
    mock_results = [
        {
            'title': 'OpenAI Releases GPT-5',
            'url': 'https://example.com/1',
            'description': 'OpenAI announced GPT-5 with major improvements',
            'age': '1 day ago'
        },
        {
            'title': 'Tech News Today',
            'url': 'https://example.com/2',
            'description': 'Various tech companies announce earnings',
            'age': '2 days ago'
        },
        {
            'title': 'Anthropic and Google Partnership',
            'url': 'https://example.com/3',
            'description': 'Anthropic partners with Google Cloud for AI infrastructure',
            'age': '3 days ago'
        },
    ]

    entities = ['OpenAI', 'Anthropic', 'Stripe']

    filtered = searcher._filter_and_score_results(mock_results, entities)

    print(f"\nOriginal results: {len(mock_results)}")
    print(f"Filtered results: {len(filtered)}")
    print("\nFiltered Results:")
    for idx, result in enumerate(filtered, 1):
        print(f"\n{idx}. {result['title']}")
        print(f"   Matched entities: {result.get('matched_entities', [])}")
        print(f"   Relevance score: {result.get('relevance_score', 0):.2f}")

if __name__ == "__main__":
    print("Context-Aware Search System - Test Suite")
    print("="*60)

    try:
        test_intent_detection()
        test_context_building()
        test_result_filtering()

        print("\n\n" + "="*60)
        print("All tests completed!")
        print("\nNext steps:")
        print("1. Run a real search: python search.py \"portfolio updates\"")
        print("2. Try industry search: python search.py \"AI industry trends\"")
        print("3. Try company search: python search.py \"news about [company]\"")

    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
