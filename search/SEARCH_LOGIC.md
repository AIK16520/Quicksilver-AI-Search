# Intelligent Search System - How It Works

## Overview

The search system uses a **3-tier priority detection** system:

1. **PRIORITY 1**: Check if query mentions a specific portfolio company → **Company Mode**
2. **PRIORITY 2**: Check if query is about industry/technology/business trends → **Industry Mode**
3. **DEFAULT**: General portfolio monitoring → **Portfolio Mode**

This ensures you ALWAYS get relevant results about companies you care about, never generic web results.

---

## Detection Logic

### Priority 1: Company Mode
**Triggers when:** Query explicitly mentions a portfolio company name

**Examples:**
- ✅ "news about Stripe"
- ✅ "what is OpenAI doing"
- ✅ "Anthropic latest developments"
- ✅ "tell me about [any portfolio company]"

**What it does:**
1. Searches for the specific company + its competitors
2. Uses company's keywords for context
3. Filters results to ONLY include mentions of these entities
4. Focuses insights on competitive intelligence

**Search expansion:**
```
Original: "news about Stripe"
Expanded: "news about Stripe (Stripe OR Square OR Adyen OR Braintree)"
```

**Filtering:** STRICT - Must mention company or competitor

---

### Priority 2: Industry Mode
**Triggers when:** Query contains trend/industry keywords but NO specific company

**Trend keywords:**
- industry, sector, market, space, landscape, ecosystem
- trends, developments, happening in, new in
- technology, innovations, emerging
- business model, funding rounds, M&A, acquisitions
- startup, startups, ipo, valuations

**Examples:**
- ✅ "AI infrastructure trends"
- ✅ "what's happening in fintech"
- ✅ "vertical SaaS market landscape"
- ✅ "emerging technologies in payments"
- ✅ "startup funding rounds this week"

**What it does:**
1. Identifies which portfolio companies are related to this topic
2. Collects their competitors
3. Gathers all related keywords
4. Searches for: topic + companies + competitors + keywords
5. Filters for articles mentioning entities OR keywords

**Search expansion:**
```
Original: "AI infrastructure trends"
Expanded: "AI infrastructure trends (GPUs OR cloud computing OR ML platforms OR OpenAI OR Anthropic...)"
```

**Filtering:** LENIENT - Entity OR keyword match (to discover new trends)

---

### Default: Portfolio Mode
**Triggers when:** No specific company/industry detected, OR explicit portfolio request

**Portfolio keywords:**
- portfolio, our investments, our companies
- all companies, portfolio companies, our portcos

**Examples:**
- ✅ "portfolio updates"
- ✅ "any news on our companies"
- ✅ "latest news" (vague query)
- ✅ "what's new" (vague query)
- ✅ "any updates" (vague query)

**What it does:**
1. Monitors ALL portfolio companies
2. Collects ALL keywords from all companies
3. Searches for: companies + top keywords
4. Filters for articles mentioning companies OR multiple keywords

**Search expansion:**
```
Original: "latest news"
Expanded: "latest news (Company1 OR Company2 OR Company3 OR AI OR fintech OR SaaS OR payments...)"
```

**Filtering:** BALANCED - Entity mention OR 2+ keyword matches (reduces noise)

---

## Smart Context Building

### For Company Mode:
```json
{
  "entities": ["Stripe", "Square", "Adyen", "Braintree"],
  "keywords": ["payments", "embedded finance", "API"],
  "focus": "Stripe"
}
```

### For Industry Mode:
```json
{
  "entities": ["OpenAI", "Anthropic", "Cohere", "Google AI", "Microsoft AI"],
  "keywords": ["AI infrastructure", "GPUs", "LLMs", "cloud computing"],
  "topic": "AI infrastructure"
}
```

### For Portfolio Mode:
```json
{
  "entities": ["All 15 portfolio companies"],
  "keywords": ["Top 20 unique keywords from all companies"],
  "monitoring": "Full portfolio"
}
```

---

## Intelligent Filtering & Scoring

### Company Mode - STRICT
**Rule:** Must mention the company or a competitor

**Scoring:**
- Base: 0.3 per entity mention (max 0.7)
- Bonus: +0.3 if entity in title

**Why:** You asked about a specific company, so results MUST be about that company.

### Industry Mode - LENIENT
**Rule:** Entity OR keyword match

**Scoring:**
- Entities: 0.2 per entity (max 0.5)
- Keywords: 0.15 per keyword (max 0.4)
- Bonus: +0.2 if entity in title, +0.1 if keyword in title

**Why:** Discovering trends requires broader filtering. A relevant article about "AI infrastructure funding" might not mention your portfolio companies directly.

### Portfolio Mode - BALANCED
**Rule:** Entity mention OR 2+ keyword matches

**Scoring:**
- Entities: 0.3 per entity (max 0.7)
- Keywords: 0.2 per keyword (max 0.6, requires 2+)
- Bonus: +0.2 if entity in title

**Why:** Balance between finding all portfolio mentions AND discovering relevant industry trends, while avoiding noise.

---

## Query Examples with Expected Behavior

### Scenario 1: Specific Company
```
Query: "news about Stripe"
Mode: company_mode
Entity: Stripe
Search: Stripe + competitors
Filter: MUST mention Stripe or competitor
Result: 10 articles specifically about Stripe/Square/Adyen
```

### Scenario 2: Industry Research
```
Query: "AI infrastructure trends"
Mode: industry_mode
Topic: AI infrastructure
Search: AI infrastructure + related companies + keywords
Filter: Entity OR keyword match
Result: Mix of articles about companies, funding, tech developments
```

### Scenario 3: Vague Query
```
Query: "what's new"
Mode: portfolio_mode (default)
Search: All portfolio companies + top keywords
Filter: Entity OR 2+ keywords
Result: Important updates about your portfolio companies
```

### Scenario 4: Explicit Portfolio Request
```
Query: "portfolio updates"
Mode: portfolio_mode
Search: All companies + keywords
Filter: Entity OR 2+ keywords
Result: Comprehensive portfolio monitoring results
```

### Scenario 5: Technology Trend (No Portfolio Match)
```
Query: "quantum computing startups"
Mode: industry_mode
Topic: quantum computing startups
Search: quantum + computing + startups + any related portfolio companies
Filter: Entity OR keyword match
Result: General quantum computing news + any portfolio company mentions
```

---

## Why This Design?

### Problem with Old Approach:
- ❌ Fuzzy matching caused false positives ("open" → "OpenAI")
- ❌ Generic queries returned irrelevant web results
- ❌ No way to discover industry trends without company mentions

### Solution - Priority-Based Detection:
1. ✅ **Precise company matching** - Only exact matches, no false positives
2. ✅ **Always portfolio-relevant** - Default to monitoring your companies
3. ✅ **Flexible trend discovery** - Industry mode allows broader exploration
4. ✅ **Smart filtering** - Mode-specific rules balance precision vs discovery

---

## Tips for Best Results

### For Company Research:
✅ **DO:** "news about Stripe"
✅ **DO:** "Stripe latest developments"
❌ **DON'T:** "payment companies" (too vague, triggers industry mode)

### For Industry Trends:
✅ **DO:** "AI infrastructure trends"
✅ **DO:** "fintech funding rounds"
✅ **DO:** "emerging technologies in SaaS"
❌ **DON'T:** Include company names if you want broad trends

### For Portfolio Monitoring:
✅ **DO:** "portfolio updates"
✅ **DO:** "latest news" (vague queries default to portfolio)
✅ **DO:** "any updates on our companies"
❌ **DON'T:** Be too specific if you want to monitor everything

---

## Testing Your Queries

Run the test script to see how queries are interpreted:
```bash
python test_context_search.py
```

Or check logs when running searches:
```bash
python search.py "your query here"
```

Look for lines like:
```
INFO:search:Detected: company_mode for 'Stripe' (exact word match)
INFO:search:Search context: Company: Stripe + 3 competitors
INFO:search:Company mode: Expanded with 4 companies
INFO:search:Filtered 20 → 12 relevant results
```
