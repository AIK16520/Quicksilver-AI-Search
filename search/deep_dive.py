# deep_dive.py

"""
Deep dive search on specific entities (companies, partnerships, technologies)
Provides focused, detailed analysis on a single entity
"""

import logging
import requests
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from openai import OpenAI
from core.config import OPENAI_API_KEY, BRAVE_API_KEY, GPT_MODEL

logger = logging.getLogger("deep_dive")

@dataclass
class DeepDiveReport:
    """Report from a deep dive on a specific entity"""
    entity: str
    entity_type: str  # "company", "partnership", "technology"
    context: str  # Original search context
    generated_at: str
    
    # Core information
    overview: str
    key_facts: List[str]
    recent_developments: List[Dict[str, Any]]
    
    # Entity-specific sections
    competitors: Optional[List[str]] = None  # For companies
    technologies_used: Optional[List[str]] = None  # For companies
    business_model: Optional[str] = None  # For companies
    partners: Optional[List[str]] = None  # For partnerships
    companies_using: Optional[List[str]] = None  # For technologies
    
    # Related entities to explore
    related_entities: List[Dict[str, Any]] = None
    suggested_next_searches: List[str] = None


class DeepDiveService:
    """
    Performs focused deep dive analysis on specific entities
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.brave_key = BRAVE_API_KEY
        self.last_request_time = 0
        self.min_request_interval = 0.5
    
    def deep_dive(
        self,
        entity: str,
        entity_type: str,
        context: str = "",
        max_articles: int = 15
    ) -> DeepDiveReport:
        """
        Perform deep dive on a specific entity
        
        Args:
            entity: The entity to research (e.g., "Refinitiv", "Goldman Sachs")
            entity_type: Type of entity ("company", "partnership", "technology")
            context: Original search context for relevance
            max_articles: Maximum articles to analyze
            
        Returns:
            DeepDiveReport with focused analysis
        """
        logger.info(f"Deep diving on {entity} ({entity_type})")
        
        # Generate targeted search queries
        queries = self._generate_deep_dive_queries(entity, entity_type, context)
        
        # Gather information
        articles = []
        for query in queries:
            results = self._brave_search(query, num_results=5)
            articles.extend(results)
            time.sleep(self.min_request_interval)
        
        articles = articles[:max_articles]
        
        # Extract information
        if self.openai_client:
            analysis = self._ai_analysis(entity, entity_type, context, articles)
        else:
            analysis = self._basic_analysis(entity, entity_type, articles)
        
        # Build report
        report = DeepDiveReport(
            entity=entity,
            entity_type=entity_type,
            context=context,
            generated_at=datetime.now().isoformat(),
            overview=analysis.get('overview', ''),
            key_facts=analysis.get('key_facts', []),
            recent_developments=analysis.get('recent_developments', []),
            competitors=analysis.get('competitors'),
            technologies_used=analysis.get('technologies_used'),
            business_model=analysis.get('business_model'),
            partners=analysis.get('partners'),
            companies_using=analysis.get('companies_using'),
            related_entities=analysis.get('related_entities', []),
            suggested_next_searches=analysis.get('suggested_next_searches', [])
        )
        
        logger.info(f"Deep dive complete on {entity}")
        return report
    
    def _generate_deep_dive_queries(
        self,
        entity: str,
        entity_type: str,
        context: str
    ) -> List[str]:
        """Generate targeted queries for deep dive"""
        
        queries = []
        
        if entity_type == "company":
            queries = [
                f'"{entity}" company overview',
                f'"{entity}" latest news 2024 2025',
                f'"{entity}" competitors',
                f'"{entity}" technology stack',
                f'"{entity}" business model',
                f'"{entity}" funding partnerships',
            ]
            
            # Add context-specific query
            if context:
                queries.append(f'"{entity}" {context[:50]}')
        
        elif entity_type == "partnership":
            # For "Goldman Sachs partnered with AI startups"
            queries = [
                entity,  # The full partnership description
                f'{entity} details',
                f'{entity} announcement',
                f'{entity} 2024 2025'
            ]
        
        elif entity_type == "technology":
            queries = [
                f'"{entity}" technology overview',
                f'companies using "{entity}"',
                f'"{entity}" use cases',
                f'"{entity}" in {context}' if context else f'"{entity}" applications'
            ]
        
        elif entity_type == "vague_mention":
            # This is a resolution query for "several AI startups"
            queries = [entity]  # entity is already the resolution query
        
        return queries
    
    def _brave_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform Brave search with rate limiting"""
        if not self.brave_key:
            return []
        
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_key
        }
        params = {
            "q": query,
            "count": min(num_results, 10),
            "result_filter": "web"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            self.last_request_time = time.time()
            results = data.get("web", {}).get("results", [])
            
            return [{
                'title': r.get('title', ''),
                'url': r.get('url', ''),
                'description': r.get('description', ''),
                'query': query
            } for r in results]
            
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            self.last_request_time = time.time()
            return []
    
    def _ai_analysis(
        self,
        entity: str,
        entity_type: str,
        context: str,
        articles: List[Dict]
    ) -> Dict:
        """Use AI to analyze gathered information"""
        
        # Compile article information
        articles_text = "\n\n".join([
            f"Title: {a['title']}\nURL: {a['url']}\nDescription: {a['description']}"
            for a in articles[:10]
        ])
        
        if entity_type == "company":
            prompt = f"""Analyze this company based on the search results:

Company: {entity}
Context: {context}

Search Results:
{articles_text}

Provide a focused analysis. Return ONLY valid JSON:
{{
  "overview": "2-3 sentence overview of what the company does",
  "key_facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"],
  "recent_developments": [
    {{"date": "2024", "description": "what happened"}},
    ...
  ],
  "competitors": ["Company A", "Company B", "Company C"],
  "technologies_used": ["Tech 1", "Tech 2", "Tech 3"],
  "business_model": "How they make money",
  "related_entities": [
    {{"name": "Related Entity", "type": "competitor", "relevance": "why relevant"}},
    ...
  ],
  "suggested_next_searches": [
    "Specific thing to search for next",
    ...
  ]
}}

Focus on information from the search results. Be specific and actionable."""
        
        elif entity_type == "partnership":
            prompt = f"""Analyze this partnership/relationship based on search results:

Partnership: {entity}
Context: {context}

Search Results:
{articles_text}

Return ONLY valid JSON:
{{
  "overview": "What is this partnership about",
  "key_facts": ["fact 1", "fact 2", "fact 3"],
  "partners": ["Partner Company 1", "Partner Company 2"],
  "recent_developments": [
    {{"date": "2024", "description": "partnership details"}},
    ...
  ],
  "related_entities": [
    {{"name": "Related Entity", "type": "company", "relevance": "why relevant"}},
    ...
  ],
  "suggested_next_searches": ["thing to search next", ...]
}}"""
        
        else:  # technology or vague_mention
            prompt = f"""Analyze this technology/topic based on search results:

Topic: {entity}
Context: {context}

Search Results:
{articles_text}

Return ONLY valid JSON:
{{
  "overview": "What is this technology/topic",
  "key_facts": ["fact 1", "fact 2", "fact 3"],
  "companies_using": ["Company 1", "Company 2", "Company 3"],
  "recent_developments": [
    {{"date": "2024", "description": "what happened"}},
    ...
  ],
  "related_entities": [
    {{"name": "Related Entity", "type": "company", "relevance": "why relevant"}},
    ...
  ],
  "suggested_next_searches": ["thing to search next", ...]
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a research analyst providing focused analysis. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            import json
            analysis = json.loads(response.choices[0].message.content.strip())
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._basic_analysis(entity, entity_type, articles)
    
    def _basic_analysis(
        self,
        entity: str,
        entity_type: str,
        articles: List[Dict]
    ) -> Dict:
        """Fallback basic analysis without AI"""
        
        key_facts = []
        for article in articles[:5]:
            if article.get('description'):
                key_facts.append(article['description'][:100])
        
        recent_developments = [{
            'date': '2024',
            'description': article.get('title', '')
        } for article in articles[:3]]
        
        return {
            'overview': f"Analysis of {entity} based on {len(articles)} articles",
            'key_facts': key_facts,
            'recent_developments': recent_developments,
            'related_entities': [],
            'suggested_next_searches': [
                f"{entity} competitors",
                f"{entity} latest news"
            ]
        }

