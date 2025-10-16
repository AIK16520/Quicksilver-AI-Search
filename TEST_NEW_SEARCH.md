# Testing the New Interactive Search

## Problem: Still Seeing Portfolio References?

You're likely calling the **OLD** endpoint which still has portfolio-focused prompts.

## Quick Fix

### ❌ DON'T USE (Old):
```bash
POST http://localhost:8000/search
{
  "query": "hedge fund AI analysis...",
  "include_web": true,
  "limit": 10,
  "generate_insights": true
}
```
**This returns:** Portfolio company implications, VC team should follow up, etc.

---

### ✅ USE THIS (New):
```bash
POST http://localhost:8000/market-intelligence
{
  "query": "I have a company that does hedge fund and trade analysis using AI. It has agents that scrape market data, pricing data, and financial news and gives funds automated updates if anything changes. Give me updates in the space.",
  "max_results_per_dimension": 8,
  "include_ai_insights": true,
  "format_type": "api"
}
```
**This returns:** Pure market analysis with interactive deep dive buttons, NO portfolio references.

---

## Testing in Frontend

### Old Way (Still Has Portfolio References):
```javascript
// DON'T USE
const result = await searchAPI.search(query, {
  includeWeb: true,
  limit: 10,
  generateInsights: true
});
```

### New Way (No Portfolio References):
```javascript
// USE THIS
const result = await searchAPI.analyzeMarketIntelligence(query, {
  maxResultsPerDimension: 8,
  includeAiInsights: true,
  formatType: "api"
});

// Now you get interactive elements
const companies = result.data.interactive_elements.companies;
companies.forEach(company => {
  console.log(`${company.name} [${company.actions[0].label}]`);
});
```

---

## Response Comparison

### OLD Endpoint Response:
```json
{
  "insights": {
    "summary": "1. PORTFOLIO UPDATES\n- Currently, there are no specific...\n\n2. COMPETITIVE UPDATES\n- Kavout recently announced...\n\n5. PORTFOLIO COMPANY IMPLICATIONS\n- BitGo should monitor...\n- The VC team should follow up..."
  }
}
```

### NEW Endpoint Response:
```json
{
  "data": {
    "executive_summary": "The hedge fund AI space is rapidly evolving with companies like Refinitiv, Bloomberg, and emerging startups...",
    
    "key_players": [
      {"name": "Refinitiv", "description": "Financial data provider..."},
      {"name": "Bloomberg", "description": "Terminal with AI analytics..."}
    ],
    
    "emerging_trends": [
      "Real-time sentiment analysis from news",
      "AI-powered trading algorithms",
      "Regulatory compliance automation"
    ],
    
    "recommended_next_steps": [
      "Deep dive on Refinitiv's data scraping capabilities",
      "Research Goldman Sachs AI partnerships",
      "Analyze pricing models in financial data space"
    ],
    
    "interactive_elements": {
      "companies": [
        {
          "name": "Refinitiv",
          "context": "announced enhancements to data scraping",
          "actions": [
            {
              "type": "deep_dive",
              "label": "Deep Dive on Refinitiv",
              "endpoint": "/deep-dive",
              "params": {
                "entity": "Refinitiv",
                "entity_type": "company"
              }
            }
          ]
        }
      ],
      "vague_mentions": [
        {
          "text": "several AI startups",
          "resolution_query": "Goldman Sachs AI partnerships 2024",
          "actions": [...]
        }
      ]
    }
  }
}
```

---

## Python Testing

```python
from search.search import SearchService

service = SearchService()

# OLD (has portfolio references)
old_result = service.search(
    query="hedge fund AI analysis companies",
    include_web=True,
    limit=10,
    generate_insights=True
)
print(old_result['insights']['summary'])
# Output: "PORTFOLIO COMPANY IMPLICATIONS - BitGo should monitor..."

# NEW (no portfolio references, interactive)
new_result = service.analyze_market_intelligence(
    query="I have a company that does hedge fund and trade analysis using AI...",
    max_results_per_dimension=8,
    include_ai_insights=True,
    format_type="api"
)
print(new_result['executive_summary'])
# Output: "The hedge fund AI space is rapidly evolving..."

print(new_result['interactive_elements']['companies'][0])
# Output: {"name": "Refinitiv", "actions": [...]}
```

---

## API Endpoint List

| Endpoint | Purpose | Portfolio References? | Interactive Elements? |
|----------|---------|----------------------|----------------------|
| `/search` | Legacy search | ✅ YES (old) | ❌ NO |
| `/market-intelligence` | New multi-dimensional | ❌ NO | ✅ YES |
| `/deep-dive` | Focused entity analysis | ❌ NO | ✅ YES |

---

## How to Switch

### Backend (if calling directly):
```python
# Change this:
result = service.search(query)

# To this:
result = service.analyze_market_intelligence(query)
```

### Frontend (React/JS):
```javascript
// Change this:
const result = await searchAPI.search(query);

// To this:
const result = await searchAPI.analyzeMarketIntelligence(query);
```

### API Calls:
```bash
# Change this:
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "..."}'

# To this:
curl -X POST http://localhost:8000/market-intelligence \
  -H "Content-Type: application/json" \
  -d '{"query": "..."}'
```

---

## Verify You're Using the Right Endpoint

Check the response structure:
- ✅ If you see `data.interactive_elements` → You're using the NEW endpoint ✓
- ❌ If you see `insights.summary` with "PORTFOLIO UPDATES" → You're using the OLD endpoint

---

## Quick Test Command

```bash
# Test the NEW endpoint
curl -X POST http://localhost:8000/market-intelligence \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I have a company that does hedge fund and trade analysis using AI. Give me updates in the space.",
    "max_results_per_dimension": 8,
    "include_ai_insights": true,
    "format_type": "api"
  }' | jq '.data.executive_summary'

# Should NOT mention portfolio companies
```

If you're still seeing portfolio references, you're calling `/search` instead of `/market-intelligence`.

