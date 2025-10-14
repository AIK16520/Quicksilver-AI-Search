# Context-Aware Search System

## Overview

The search system now intelligently detects query intent and automatically tailors results based on whether you're asking about:
- **Portfolio companies** - Your investments
- **Industry trends** - Market sectors and competitive landscape
- **Specific companies** - Individual companies or competitors

## How It Works

### 1. Intent Detection

The system automatically classifies your query into one of three modes:

#### Portfolio Mode
Triggered by keywords like: "portfolio", "our investments", "our companies", "portfolio companies"

**Example queries:**
```
python search.py "any news on our portfolio companies"
python search.py "updates about our investments"
```

**What it does:**
- Fetches ALL portfolio companies from database
- Searches for mentions of any portfolio company
- Focuses insights on direct portfolio impact

#### Industry Mode
Triggered by keywords like: "industry", "sector", "space", "market", "landscape", "trends in"

**Example queries:**
```
python search.py "what's happening in AI infrastructure industry"
python search.py "fintech sector trends"
python search.py "vertical SaaS market landscape"
```

**What it does:**
- Matches industry to portfolio companies
- Includes both portfolio companies AND their competitors
- Provides competitive landscape analysis
- Focuses on market dynamics and investment implications

#### Company Mode (Default)
Triggered by specific company names or as fallback

**Example queries:**
```
python search.py "news about OpenAI"
python search.py "Stripe latest developments"
python search.py "what is Anthropic working on"
```

**What it does:**
- Fuzzy matches company name to portfolio
- Includes the company + its competitors in search
- Provides competitive intelligence
- Shows implications for portfolio if relevant

### 2. Dynamic Query Expansion

Based on the detected mode, the system automatically expands your Brave search query:

**Portfolio Mode:**
```
Original: "portfolio company news"
Expanded: "portfolio company news (Company1 OR Company2 OR Company3...)"
```

**Industry Mode:**
```
Original: "AI industry trends"
Expanded: "AI industry trends (PortfolioCompany1 OR Competitor1 OR Competitor2...)"
```

**Company Mode:**
```
Original: "OpenAI news"
Expanded: "OpenAI news (OpenAI OR Anthropic OR Google AI...)"
```

### 3. Smart Result Filtering

Web results are filtered to only show articles that mention relevant entities:

- **Relevance scoring** based on entity mentions
- **Title boost** for companies mentioned in headlines
- **Entity tracking** to show which companies are mentioned
- **Automatic sorting** by relevance score

### 4. Context-Aware Insights

AI-generated insights adapt to the query mode:

**Portfolio Mode Focus:**
- Portfolio updates (which companies mentioned)
- Key developments affecting investments
- Recommended actions for VC team

**Industry Mode Focus:**
- Executive summary of industry trends
- Competitive landscape analysis
- Market dynamics (funding, M&A, partnerships)
- Investment implications and strategy

**Company Mode Focus:**
- Company overview and major moves
- Key developments (product, funding, partnerships)
- Implications for portfolio/competition
- Investment considerations

## Example Usage

### Example 1: Portfolio Monitoring
```bash
python search.py "any updates on our portfolio companies"
```

**Output:**
```
Mode: Portfolio Mode
Context: All 15 portfolio companies

AI-POWERED INSIGHTS:
1. PORTFOLIO UPDATES
   - Company X raised Series B ($50M)
   - Company Y launched new product

2. KEY DEVELOPMENTS
   - Competitive threat from Company Z in fintech space
   - Partnership opportunity identified

3. RECOMMENDED ACTIONS
   - Schedule check-in with Company X
   - Monitor Company Z's product launch
```

### Example 2: Industry Research
```bash
python search.py "what's happening in vertical SaaS industry"
```

**Output:**
```
Mode: Industry Mode
Context: Industry: vertical SaaS (5 portfolio companies, 12 competitors)

AI-POWERED INSIGHTS:
1. EXECUTIVE SUMMARY
   - Consolidation wave in vertical SaaS
   - AI integration becoming table stakes

2. COMPETITIVE LANDSCAPE
   - Major players: [Portfolio companies vs competitors]
   - New entrant threat from horizontal platforms

3. MARKET DYNAMICS
   - $2B in M&A activity this quarter
   - Average Series B valuations down 20%
```

### Example 3: Company Intelligence
```bash
python search.py "news about Stripe"
```

**Output:**
```
Mode: Company Mode
Context: Company: Stripe + 3 competitors

AI-POWERED INSIGHTS:
1. COMPANY OVERVIEW
   - Stripe launched embedded finance platform
   - Expanded to 15 new markets

2. KEY DEVELOPMENTS
   - Product: New crypto on-ramp feature
   - Competitive: Undercutting Square on pricing

3. IMPLICATIONS
   - Threat to portfolio company PaymentCo
   - Consider investment in embedded finance
```

## Advanced Features

### Fuzzy Company Matching
The system uses fuzzy matching to find companies even with typos:
```bash
python search.py "news about antropic"  # Matches "Anthropic"
python search.py "what is openai doing"  # Matches "OpenAI"
```

### Automatic Competitor Tracking
When you search for a company, competitors are automatically included:
- Searches for the company AND its competitors
- Shows competitive positioning
- Identifies threats and opportunities

### Smart Entity Limiting
For queries with many entities (e.g., all portfolio companies):
- Limits to 20 entities to avoid query overload
- Prioritizes portfolio companies over competitors
- Still filters results for relevance

## Configuration

### Portfolio Company Setup
Ensure your `portfolio_companies` table has these fields:
- `company_name` (required)
- `competitors` (array, optional but recommended)
- `industry` (string, optional but recommended)
- `keywords` (array, optional)
- `what_they_build` (string, for semantic matching)

### Example Portfolio Entry
```json
{
  "company_name": "Stripe",
  "industry": "Fintech",
  "competitors": ["Square", "Adyen", "Braintree"],
  "keywords": ["payments", "embedded finance", "API"],
  "what_they_build": "Payment processing infrastructure for online businesses"
}
```

## Tips for Best Results

1. **Be natural** - Use conversational queries like "what's happening in fintech"
2. **Use keywords** - Include "portfolio", "industry", or company names for better detection
3. **Check the mode** - The displayed mode shows how your query was interpreted
4. **Review entities** - The context shows which companies are being tracked
5. **Refine if needed** - If wrong mode detected, rephrase your query

## API Usage

```python
from search import SearchService

searcher = SearchService()

# Will auto-detect mode
results = searcher.search(
    query="portfolio company updates",
    include_web=True,
    limit=10,
    generate_insights=True
)

# Check detected mode
print(f"Mode: {results['mode']}")
print(f"Context: {results['context']['description']}")

# Access filtered results
for item in results['web']:
    print(f"{item['title']}")
    print(f"Mentions: {item.get('matched_entities', [])}")
    print(f"Relevance: {item.get('relevance_score', 0):.1%}")
```

## Troubleshooting

**Problem:** Wrong mode detected
- **Solution:** Use explicit keywords ("portfolio companies", "industry", company name)

**Problem:** No results found
- **Solution:** Check that portfolio companies are in database with correct names

**Problem:** Too many/few results
- **Solution:** Adjust `--limit` parameter or refine query

**Problem:** Entities not being matched
- **Solution:** Ensure `competitors` field is populated in portfolio_companies table
