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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Quicksilver Search API",
        "version": "1.0.0",
        "description": "VC-focused search service with portfolio intelligence",
        "endpoints": {
            "search": "/search (POST) - Perform intelligent search",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
