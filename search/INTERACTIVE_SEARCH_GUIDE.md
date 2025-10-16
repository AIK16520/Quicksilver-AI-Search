# Interactive Search Guide

## Overview

The interactive search system allows you to **explore mentioned entities by drilling down** into them. Instead of getting a static report, you now get:

1. **Clickable entities** - Companies, partnerships, technologies mentioned in reports
2. **Deep dive buttons** - Click to get focused analysis on specific entities
3. **Vague entity resolution** - Automatically find specifics when reports mention "several AI startups"
4. **NO portfolio references** - Pure market analysis without investment recommendations

## Problem Solved

### Before (Old Behavior):
```
Report says: "Goldman Sachs partnered with several AI startups"
You: "Which startups?"
System: 🤷 (you have to manually search)
```

### After (New Behavior):
```
Report says: "Goldman Sachs partnered with several AI startups" [🔍 Find Partners]
You: *click button*
System: Returns list of specific startups: Kensho, Symphony, etc.
```

## How It Works

### Step 1: Market Intelligence Search
When you run a market intelligence query, the system now:

1. Performs multi-dimensional analysis (as before)
2. **Extracts entities** from results using AI
3. **Creates interactive elements** for each entity
4. Returns them in `interactive_elements` section

### Step 2: Interactive Elements in Response

Example response structure:

```json
{
  "query": "hedge fund AI analysis companies",
  "executive_summary": "...",
  "key_players": [...],
  
  "interactive_elements": {
    "companies": [
      {
        "name": "Refinitiv",
        "context": "announced enhancements to data scraping",
        "mentioned_in": ["competitive", "technology"],
        "confidence": 0.85,
        "actions": [
          {
            "type": "deep_dive",
            "label": "Deep Dive on Refinitiv",
            "endpoint": "/deep-dive",
            "params": {
              "entity": "Refinitiv",
              "entity_type": "company",
              "context": "hedge fund AI analysis"
            }
          },
          {
            "type": "recent_news",
            "label": "Recent News",
            "endpoint": "/search",
            "params": {
              "query": "Refinitiv latest news developments"
            }
          },
          {
            "type": "competitors",
            "label": "Find Competitors",
            "endpoint": "/analyze-competitors",
            "params": {
              "query": "Refinitiv"
            }
          }
        ]
      }
    ],
    
    "partnerships": [
      {
        "description": "Goldman Sachs partnered with AI startups",
        "context": "...",
        "vague": false,
        "actions": [
          {
            "type": "resolve_partnership",
            "label": "Find Partnership Details",
            "endpoint": "/deep-dive",
            "params": {
              "entity": "Goldman Sachs AI startup partnerships",
              "entity_type": "partnership",
              "context": "..."
            }
          }
        ]
      }
    ],
    
    "vague_mentions": [
      {
        "text": "several AI startups",
        "resolution_query": "Goldman Sachs AI startup partnerships 2024",
        "actions": [
          {
            "type": "resolve",
            "label": "Find several AI startups",
            "query": "Goldman Sachs AI startup partnerships 2024"
          }
        ]
      }
    ]
  }
}
```

### Step 3: Deep Dive

When you click a deep dive button, it calls:

```javascript
// Frontend
const result = await searchAPI.deepDive("Refinitiv", {
  entityType: "company",
  context: "hedge fund AI analysis",
  maxArticles: 15
});
```

**Deep Dive Response:**

```json
{
  "data": {
    "entity": "Refinitiv",
    "entity_type": "company",
    "overview": "Refinitiv is a global provider of financial market data...",
    "key_facts": [
      "Owned by London Stock Exchange Group",
      "Serves 400,000+ financial professionals",
      "Provides real-time market data and analytics"
    ],
    "recent_developments": [
      {
        "date": "2024",
        "description": "Launched AI-powered real-time analytics platform"
      },
      {
        "date": "2024",
        "description": "Enhanced data scraping capabilities"
      }
    ],
    "competitors": ["Bloomberg", "FactSet", "S&P Global"],
    "technologies_used": ["AI/ML", "Real-time data processing", "Cloud infrastructure"],
    "business_model": "Subscription-based SaaS + data licensing",
    "related_entities": [
      {
        "name": "Bloomberg",
        "type": "competitor",
        "relevance": "Direct competitor in financial data"
      }
    ],
    "suggested_next_searches": [
      "Refinitiv vs Bloomberg comparison",
      "Refinitiv pricing model",
      "Refinitiv API documentation"
    ]
  }
}
```

## API Endpoints

### 1. Market Intelligence (Enhanced)

**POST** `/market-intelligence`

Now includes `interactive_elements` in response.

```javascript
const result = await searchAPI.analyzeMarketIntelligence(
  "hedge fund AI analysis companies"
);

// Access interactive elements
const companies = result.data.interactive_elements.companies;
companies.forEach(company => {
  console.log(`${company.name} - ${company.actions[0].label}`);
});
```

### 2. Deep Dive (NEW)

**POST** `/deep-dive`

Request:
```json
{
  "entity": "Refinitiv",
  "entity_type": "company",
  "context": "hedge fund AI",
  "max_articles": 15
}
```

```javascript
const result = await searchAPI.deepDive("Refinitiv", {
  entityType: "company",
  context: "hedge fund AI",
  maxArticles: 15
});
```

## Entity Types

### 1. Companies
- **Extracted from**: Key players, competitive analysis, article mentions
- **Actions**: Deep Dive, Recent News, Find Competitors
- **Example**: "Refinitiv", "Bloomberg", "Goldman Sachs"

### 2. Partnerships
- **Extracted from**: Mentions of collaborations, acquisitions
- **Actions**: Find Partnership Details, Deep Dive on Partners
- **Example**: "Goldman Sachs partnered with AI startups"

### 3. Technologies
- **Extracted from**: Tech stack mentions, tool references
- **Actions**: Explore Technology, Companies Using This
- **Example**: "AI agents", "real-time data processing"

### 4. Vague Mentions (Special)
- **Detected**: "several AI startups", "multiple companies", "various players"
- **Resolution**: Auto-generates search query to find specific entities
- **Example**: "several AI startups" → searches for "Goldman Sachs AI startup partnerships 2024"

## No More Portfolio References

The AI prompts have been updated to:

❌ **REMOVED:**
- "Portfolio company implications"
- "VC team should follow up with..."
- Investment recommendations
- "Our companies should..."

✅ **FOCUS ON:**
- Pure market analysis
- Specific company names
- Concrete trends and technologies
- Actionable research next steps

**Old Output:**
```
PORTFOLIO COMPANY IMPLICATIONS
- Aclaimant and BitGo should closely monitor Refinitiv...
- The VC team should follow up with companies...
```

**New Output:**
```
RECOMMENDED NEXT STEPS
- Deep dive on Refinitiv's new analytics platform
- Research Bloomberg's competitive response
- Analyze pricing models in the financial data space
```

## Usage Examples

### Example 1: Exploring Refinitiv

```javascript
// 1. Run market intelligence
const marketReport = await searchAPI.analyzeMarketIntelligence(
  "hedge fund AI analysis companies"
);

// 2. Find Refinitiv in interactive elements
const refinitiv = marketReport.data.interactive_elements.companies
  .find(c => c.name === "Refinitiv");

// 3. Deep dive on Refinitiv
const deepDive = await searchAPI.deepDive("Refinitiv", {
  entityType: "company",
  context: marketReport.data.query
});

console.log(deepDive.data.overview);
console.log(deepDive.data.competitors);
console.log(deepDive.data.business_model);
```

### Example 2: Resolving Vague Mentions

```javascript
// 1. Market report mentions "Goldman Sachs partnered with several AI startups"
const marketReport = await searchAPI.analyzeMarketIntelligence(
  "hedge fund AI companies"
);

// 2. Find the vague mention
const vagueMention = marketReport.data.interactive_elements.vague_mentions
  .find(v => v.text.includes("AI startups"));

// 3. Resolve it with deep dive
const partners = await searchAPI.deepDive(
  vagueMention.resolution_query,
  {
    entityType: "partnership",
    context: marketReport.data.query
  }
);

// Now you get: Kensho, Symphony, etc.
console.log(partners.data.partners);
```

### Example 3: Following Related Entities

```javascript
// 1. Deep dive on Refinitiv
const refinitivReport = await searchAPI.deepDive("Refinitiv");

// 2. Explore related competitor
const relatedCompetitor = refinitivReport.data.related_entities[0];

// 3. Deep dive on that too
const competitorReport = await searchAPI.deepDive(
  relatedCompetitor.name,
  { entityType: "company" }
);
```

## Frontend Integration

### Display Interactive Elements

```jsx
// In your React component
function MarketIntelligenceResults({ data }) {
  const handleDeepDive = async (company) => {
    const action = company.actions.find(a => a.type === 'deep_dive');
    const result = await searchAPI.deepDive(
      action.params.entity,
      {
        entityType: action.params.entity_type,
        context: action.params.context
      }
    );
    
    // Display deep dive results
    setDeepDiveData(result.data);
  };

  return (
    <div>
      <h3>Interactive Companies Found:</h3>
      {data.interactive_elements.companies.map(company => (
        <div key={company.name}>
          <h4>{company.name}</h4>
          <p>{company.context}</p>
          {company.actions.map(action => (
            <button key={action.type} onClick={() => {
              if (action.type === 'deep_dive') {
                handleDeepDive(company);
              }
            }}>
              {action.label}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}
```

## Confidence Scores

Entities include confidence scores (0-1):

- **0.9+**: Direct mention, high confidence (e.g., in key_players list)
- **0.8-0.9**: AI-extracted from clear context
- **0.6-0.8**: Pattern-matched or inferred
- **<0.6**: Low confidence, might need verification

Use confidence scores to:
- Prioritize which entities to show prominently
- Filter out low-confidence noise
- Provide visual indicators (e.g., star ratings)

## Best Practices

1. **Always use context**: Pass the original query as context to deep dive
2. **Follow suggested paths**: Use `suggested_next_searches` for natural exploration
3. **Handle vague mentions**: Automatically resolve them for better UX
4. **Show confidence visually**: Let users know which entities are high-confidence
5. **Batch requests**: Deep dive on multiple entities in parallel if needed

## Error Handling

```javascript
try {
  const result = await searchAPI.deepDive("Refinitiv");
} catch (error) {
  if (error.response?.status === 500) {
    // Server error - maybe entity not found or API limit reached
    console.error("Deep dive failed:", error.response.data.detail);
  }
}
```

## Performance

- Market Intelligence: ~30-60 seconds (multiple searches + entity extraction)
- Deep Dive: ~10-20 seconds (focused search on one entity)
- Entity Extraction: ~2-5 seconds (AI processing)

## Roadmap

Future enhancements:
- [ ] Session-based conversation history
- [ ] "Show me more about X" natural language follow-ups
- [ ] Side-by-side entity comparison
- [ ] Save favorite entities
- [ ] Entity relationship graphs
- [ ] Real-time entity updates

