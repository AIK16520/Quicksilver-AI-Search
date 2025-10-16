"""
FastAPI application for Quicksilver Search Service
Provides REST API endpoints for the VC search system
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os

# Add the parent directory to the path so we can import from search
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import supabase_client, OPENAI_API_KEY, BRAVE_API_KEY, EMBEDDING_MODEL, GPT_MODEL
from search.search import SearchService
from search.competitor_config import CompetitorDiscoveryConfig
from search.market_intelligence import MarketIntelligenceReport
from fastapi.responses import FileResponse, Response


# Check if supabase client is available
if supabase_client is None:
    print("Warning: Supabase client not available. Database features will not work.")

# Create FastAPI app
app = FastAPI(
    title="Quicksilver Search API",
    description="VC-focused search service with portfolio intelligence",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search service
search_service = SearchService()

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    include_web: bool = True
    limit: int = 10
    generate_insights: bool = True

class SearchResponse(BaseModel):
    query: str
    mode: str
    context: Dict[str, Any]
    database: List[Dict[str, Any]]
    web: List[Dict[str, Any]]
    insights: Optional[Dict[str, Any]]
    timestamp: str

class QueryTypeRequest(BaseModel):
    query: str

class QueryTypeResponse(BaseModel):
    mode: str
    entity: Optional[str]
    description: str

class ContextRequest(BaseModel):
    mode: str
    entity: Optional[str]

class ContextResponse(BaseModel):
    entities: List[str]
    keywords: List[str]
    description: str
    mode: str

class CompetitorAnalysisRequest(BaseModel):
    query: str
    include_discovery: bool = True
    config: Optional[Dict[str, Any]] = None
    search_depth: str = "moderate"

class CompetitorAnalysisResponse(BaseModel):
    query: str
    mode: str
    matched_entity: Optional[str]
    mapped_competitors: List[str]
    discovered_competitors: List[Dict[str, Any]]
    all_competitors: List[str]
    total_mapped: int
    total_discovered: int
    total_competitors: int

class MarketIntelligenceRequest(BaseModel):
    query: str
    max_results_per_dimension: int = 8
    include_ai_insights: bool = True
    format_type: str = "api"  # "api", "display", or "summary"

class MarketIntelligenceResponse(BaseModel):
    # Using flexible Dict to accommodate different format types
    data: Dict[str, Any]

class DeepDiveRequest(BaseModel):
    entity: str
    entity_type: str = "company"  # "company", "partnership", "technology"
    context: str = ""
    max_articles: int = 15

class DeepDiveResponse(BaseModel):
    data: Dict[str, Any]
@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
def favicon():
    import os
    favicon_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Quicksilver Search API",
        "version": "1.0.0",
        "description": "VC-focused search service with portfolio intelligence",
        "endpoints": {
            "search": "/search (POST) - Perform intelligent search",
            "market-intelligence": "/market-intelligence (POST) - Multi-dimensional market analysis",
            "deep-dive": "/deep-dive (POST) - Deep dive on specific company/entity",
            "analyze-competitors": "/analyze-competitors (POST) - Enhanced competitor analysis with discovery",
            "query-type": "/query-type (POST) - Detect query intent",
            "context": "/context (POST) - Build search context",
            "health": "/health (GET) - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Quicksilver Search API"}

@app.post("/query-type", response_model=QueryTypeResponse)
async def detect_query_type(request: QueryTypeRequest):
    """Detect the type and intent of a search query"""
    try:
        mode, entity = search_service._detect_query_type(request.query)
        description = ""

        if mode == "company_mode":
            description = f"Search focused on specific company: {entity}"
        elif mode == "industry_mode":
            description = f"Industry/trend analysis for: {entity}"
        elif mode == "portfolio_mode":
            description = "General portfolio monitoring across all companies"

        return QueryTypeResponse(
            mode=mode,
            entity=entity,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context", response_model=ContextResponse)
async def build_search_context(request: ContextRequest):
    """Build search context for a given mode and entity"""
    try:
        context = search_service._build_search_context(request.mode, request.entity)

        return ContextResponse(
            entities=context.get('entities', []),
            keywords=context.get('keywords', []),
            description=context.get('description', ''),
            mode=context.get('mode', request.mode)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def perform_search(request: SearchRequest):
    """Perform intelligent search with VC-focused insights"""
    try:
        results = search_service.search(
            query=request.query,
            include_web=request.include_web,
            limit=request.limit,
            generate_insights=request.generate_insights
        )

        return SearchResponse(
            query=results['query'],
            mode=results['mode'],
            context=results['context'],
            database=results['database'],
            web=results['web'],
            insights=results['insights'],
            timestamp=results['timestamp']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio-companies")
async def get_portfolio_companies():
    """Get list of all portfolio companies"""
    try:
        companies = search_service._get_portfolio_companies()
        return {"companies": companies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio-maps")
async def get_portfolio_maps():
    """Get portfolio mapping information"""
    try:
        return {
            "keywords_count": len(search_service.all_keywords),
            "industries_count": len(search_service.all_industries),
            "sample_keywords": search_service.all_keywords[:10],
            "sample_industries": search_service.all_industries[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-competitors", response_model=CompetitorAnalysisResponse)
async def analyze_competitors(request: CompetitorAnalysisRequest):
    """Enhanced competitor analysis with competitor discovery"""
    try:
        # Convert config dict to CompetitorDiscoveryConfig if provided
        config = None
        if request.config:
            config = CompetitorDiscoveryConfig(**request.config)

        results = search_service.analyze_competitors_enhanced(
            query=request.query,
            config=config,
            include_discovery=request.include_discovery
        )

        return CompetitorAnalysisResponse(
            query=results['query'],
            mode=results['mode'],
            matched_entity=results['matched_entity'],
            mapped_competitors=results['mapped_competitors'],
            discovered_competitors=results['discovered_competitors'],
            all_competitors=results['all_competitors'],
            total_mapped=results['total_mapped'],
            total_discovered=results['total_discovered'],
            total_competitors=results['total_competitors']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-intelligence", response_model=MarketIntelligenceResponse)
async def perform_market_intelligence(request: MarketIntelligenceRequest):
    """
    Perform comprehensive multi-dimensional market intelligence analysis
    
    This endpoint analyzes a market space across multiple dimensions:
    - Company landscape (key players, startups, incumbents)
    - Technology landscape (tech stack, tools, platforms)
    - Business models (pricing, monetization strategies)
    - Innovations (recent innovations, differentiators)
    - Market trends (growth, dynamics, future outlook)
    - Competitive intelligence (competitor discovery & mapping)
    
    Example query:
    "I have a company that does hedge fund and trade analysis using AI. 
     It has agents that scrape market data, pricing data, and financial news 
     and gives funds automated updates if anything changes. Give me updates in the space."
    """
    try:
        result = search_service.analyze_market_intelligence(
            query=request.query,
            max_results_per_dimension=request.max_results_per_dimension,
            include_ai_insights=request.include_ai_insights,
            format_type=request.format_type
        )
        
        return MarketIntelligenceResponse(data=result)
    except Exception as e:
        logger.error(f"Market intelligence error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deep-dive", response_model=DeepDiveResponse)
async def perform_deep_dive(request: DeepDiveRequest):
    """
    Perform deep dive on a specific entity (company, partnership, or technology)
    
    This endpoint is triggered when users click "Deep Dive" buttons in market intelligence reports.
    It provides focused, detailed analysis on a single entity.
    
    Example use cases:
    - Deep dive on "Refinitiv" after seeing it mentioned in competitive analysis
    - Find specific AI startups that "Goldman Sachs partnered with"
    - Explore a technology like "AI agents" in depth
    """
    try:
        result = search_service.deep_dive_entity(
            entity=request.entity,
            entity_type=request.entity_type,
            context=request.context,
            max_articles=request.max_articles
        )
        
        return DeepDiveResponse(data=result)
    except Exception as e:
        logger.error(f"Deep dive error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
