# Market Intelligence Search Guide

## Overview

The Market Intelligence search is an enhanced, multi-dimensional analysis system that goes beyond simple keyword search. Instead of just finding articles, it **decomposes complex queries** and analyzes market spaces across **6 key dimensions**:

1. **Company Landscape** - Key players, startups, incumbents
2. **Technology Landscape** - Tech stacks, tools, platforms
3. **Business Models** - Pricing strategies, monetization
4. **Innovations** - Recent innovations, differentiators
5. **Market Trends** - Growth dynamics, future outlook
6. **Competitive Intelligence** - Competitor discovery & mapping

## When to Use This vs. Regular Search

### Use Market Intelligence When:
- You have a **complex company/product description** to analyze
- You want to understand an **entire market space**, not just find articles
- You need insights on **competitors, technologies, and business models**
- You're doing **market research** or **competitive analysis**

**Example Query:**
```
"I have a company that does hedge fund and trade analysis using AI. 
It has agents that scrape market data, pricing data, and financial news 
and gives funds automated updates if anything changes. 
Give me updates in the space."
```

### Use Regular Search When:
- You have a **specific question** about a known company
- You want **recent news** about a portfolio company
- You're looking for **specific information**, not market overview

## How It Works

### Step 1: Query Decomposition
The system uses AI to break down your query into structured components:

```json
{
  "domain": ["hedge funds", "financial trading"],
  "technologies": ["AI agents", "web scraping", "automation"],
  "data_sources": ["market data", "pricing data", "financial news"],
  "problem_solved": "Automated analysis and alerts for hedge funds",
  "value_proposition": "Real-time automated updates on market changes",
  "keywords": ["hedge fund", "trading", "AI", "automation", "market data"],
  "industries": ["fintech", "asset management"]
}
```

### Step 2: Multi-Dimensional Search
For each dimension, the system generates targeted search queries:

**Companies Dimension:**
- "top companies in hedge funds"
- "leading hedge funds AI startups"
- "AI agents companies in hedge funds"

**Technology Dimension:**
- "AI agents in hedge funds"
- "web scraping tools for trading"
- "technology stack for hedge funds"

**Business Models Dimension:**
- "hedge funds SaaS pricing models"
- "how hedge fund companies monetize"

**Innovations Dimension:**
- "latest innovations in hedge funds"
- "new AI agents solutions for hedge funds"
- "hedge fund startup innovations 2024 2025"

**Market Trends Dimension:**
- "hedge funds market trends 2024"
- "hedge funds industry analysis"

**Competitive Intelligence:**
- Uses the competitor discovery service to find companies
- Searches for "competitors in [your space]"

### Step 3: AI-Powered Aggregation
The system synthesizes all findings into:
- **Executive Summary** - 2-3 sentence overview
- **Key Players** - 5-7 companies with descriptions
- **Emerging Trends** - 3-5 key trends
- **Recommended Next Steps** - Actionable recommendations

## API Usage

### Endpoint
```
POST /market-intelligence
```

### Request Body
```json
{
  "query": "Your detailed company/market description",
  "max_results_per_dimension": 8,
  "include_ai_insights": true,
  "format_type": "api"
}
```

**Parameters:**
- `query` (required): Natural language description of the company/space
- `max_results_per_dimension` (optional, default: 8): Articles per dimension
- `include_ai_insights` (optional, default: true): Generate AI summaries
- `format_type` (optional, default: "api"): Output format
  - `"api"` - Structured JSON for programmatic use
  - `"display"` - Formatted text for terminal display
  - `"summary"` - Condensed summary

### Response Structure (format_type: "api")

```json
{
  "data": {
    "query": "Original query",
    "generated_at": "2024-10-15T12:00:00",
    
    "analysis": {
      "domain": ["hedge funds", "trading"],
      "technologies": ["AI agents", "scraping"],
      "problem_solved": "...",
      "value_proposition": "..."
    },
    
    "executive_summary": "High-level overview...",
    "key_players": [
      {"name": "Company A", "description": "Leading AI..."},
      {"name": "Company B", "description": "Emerging startup..."}
    ],
    "emerging_trends": [
      "Trend 1: AI-powered automation...",
      "Trend 2: Real-time data integration..."
    ],
    "recommended_next_steps": [
      "Deep dive into Company A's technology",
      "Analyze pricing models in the space"
    ],
    
    "dimensions": {
      "companies": {
        "description": "Key players in the space",
        "key_findings": ["Found 15 relevant companies"],
        "companies_mentioned": ["Company A", "Company B"],
        "articles": [...],
        "article_count": 8
      },
      "technology": {...},
      "business_models": {...},
      "innovations": {...},
      "market_trends": {...},
      "competitive": {...}
    },
    
    "metadata": {
      "total_articles": 48,
      "dimensions_analyzed": 6,
      "companies_identified": 23
    }
  }
}
```

## Frontend Usage

```javascript
import searchAPI from './utils/api';

// Perform market intelligence analysis
const result = await searchAPI.analyzeMarketIntelligence(
  "Your detailed company/space description",
  {
    maxResultsPerDimension: 8,
    includeAiInsights: true,
    formatType: "api"
  }
);

// Access the results
const { data } = result;

console.log(data.executive_summary);
console.log(data.key_players);
console.log(data.dimensions.companies.companies_mentioned);
```

## Python Usage

```python
from search.search import SearchService

service = SearchService()

# Perform market intelligence analysis
report = service.analyze_market_intelligence(
    query="""I have a company that does hedge fund and trade analysis using AI. 
             It has agents that scrape market data, pricing data, and financial news 
             and gives funds automated updates if anything changes. 
             Give me updates in the space.""",
    max_results_per_dimension=8,
    include_ai_insights=True,
    format_type="api"
)

print(report['executive_summary'])
print(report['key_players'])

# Access dimensional results
companies = report['dimensions']['companies']
print(f"Found {companies['article_count']} articles")
print(f"Companies: {companies['companies_mentioned']}")
```

## Example Output

For the hedge fund AI query, you would get:

### Executive Summary
*"The hedge fund technology space is rapidly adopting AI-powered automation for market analysis and trading decisions. Leading players include established data providers like Bloomberg and emerging AI-first startups. Key trends include real-time data integration, automated decision-making, and regulatory compliance automation."*

### Key Players
1. **Bloomberg Terminal** - Established market data provider with AI features
2. **Kensho** (acquired by S&P) - AI for financial analysis
3. **Two Sigma** - Quantitative hedge fund using AI/ML
4. **AlphaSense** - AI-powered market intelligence
5. **Kavout** - AI stock analysis platform

### Emerging Trends
- AI agents replacing manual research analysts
- Real-time news sentiment analysis integration
- Automated compliance and risk monitoring
- Alternative data sources (satellite, social media)

### Dimensions Breakdown

**Companies (15 found):**
- Articles about key players
- Startup announcements
- Funding rounds

**Technology (12 articles):**
- NLP for financial news
- Agent-based systems
- Real-time data pipelines

**Business Models (8 articles):**
- SaaS subscriptions ($1K-$10K/month per fund)
- Per-trade fees
- Data licensing

**Innovations (10 articles):**
- GPT-4 for earnings call analysis
- Automated regulatory filing analysis
- Predictive market modeling

## Best Practices

1. **Be Descriptive**: Include what the company does, what tech it uses, and what problem it solves
2. **Mention Specifics**: Include data sources, customer types, key features
3. **Ask for "the space"**: Use phrases like "give me updates in the space" or "analyze the market"
4. **Review All Dimensions**: Don't just look at the summary—each dimension has valuable insights
5. **Follow Recommended Steps**: The AI suggests next steps based on findings

## Performance Notes

- Analysis takes 30-60 seconds (multiple web searches + AI processing)
- Rate-limited to avoid API throttling
- Results are cached in the response (no database storage yet)
- Best with OpenAI API key (falls back to rule-based without)

## Comparison: Regular Search vs. Market Intelligence

| Feature | Regular Search | Market Intelligence |
|---------|---------------|---------------------|
| **Input** | Simple question | Detailed description |
| **Output** | List of articles | Multi-dimensional report |
| **Analysis** | Single dimension | 6 dimensions |
| **Competitor Discovery** | Only mapped | Discovers new ones |
| **Business Model Insights** | No | Yes |
| **Technology Analysis** | No | Yes |
| **Executive Summary** | No | Yes |
| **Processing Time** | ~5 seconds | ~45 seconds |
| **Best For** | Quick updates | Market research |

## Future Enhancements

- [ ] Store reports in database for future reference
- [ ] Compare multiple market spaces side-by-side
- [ ] Track market changes over time
- [ ] Integration with Crunchbase, LinkedIn, PitchBook
- [ ] Custom dimension definitions
- [ ] Export to PDF/PowerPoint

