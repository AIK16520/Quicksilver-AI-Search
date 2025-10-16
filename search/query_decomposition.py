# query_decomposition.py

"""
Intelligent query decomposition for market intelligence searches.
Breaks down complex queries into searchable components and dimensions.
"""

import logging
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from openai import OpenAI
from core.config import OPENAI_API_KEY, GPT_MODEL

logger = logging.getLogger("query_decomposition")

@dataclass
class QueryComponents:
    """Structured representation of query components"""
    # Core business elements
    domain: List[str]  # Industry/sector (e.g., "hedge funds", "fintech")
    problem_solved: str  # What problem is being addressed
    value_proposition: str  # Key value delivered
    
    # Technology elements
    technologies: List[str]  # Tech stack mentioned (AI, agents, scraping)
    data_sources: List[str]  # Data inputs (market data, news, pricing)
    
    # Market intelligence dimensions
    search_dimensions: Dict[str, List[str]]  # Organized search queries
    
    # Keywords for discovery
    keywords: List[str]
    industries: List[str]


class QueryDecompositionService:
    """
    Decomposes complex market intelligence queries into structured components
    for comprehensive multi-dimensional search
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
    def decompose_query(self, query: str) -> QueryComponents:
        """
        Decompose a complex query into structured components
        
        Args:
            query: Natural language query describing a company/space
            
        Returns:
            QueryComponents with all extracted elements
        """
        logger.info(f"Decomposing query: {query[:100]}...")
        
        if self.openai_client:
            return self._ai_decomposition(query)
        else:
            return self._rule_based_decomposition(query)
    
    def _ai_decomposition(self, query: str) -> QueryComponents:
        """Use AI to intelligently decompose the query"""
        
        prompt = f"""Analyze this company/market description and extract structured information:

Query: {query}

IMPORTANT: Distinguish between PROVIDER and CUSTOMER industries.
- If the query says "company that does X for Y" → Y is the customer, X is what the company does
- Domain should describe the PROVIDER company's industry, NOT the customer's industry
- Target customers should be WHO USES the product/service

Examples:
1. "company that does hedge fund analysis using AI"
   - Domain: "financial analytics software", "AI platforms"
   - Target customers: "hedge funds", "asset managers"

2. "platform that helps restaurants manage orders"
   - Domain: "restaurant technology", "SaaS"
   - Target customers: "restaurants", "food service"

3. "AI tool for healthcare diagnostics"
   - Domain: "healthcare technology", "medical AI"
   - Target customers: "hospitals", "doctors"

Extract and return ONLY valid JSON with this EXACT structure:
{{
  "domain": ["provider industry 1", "provider industry 2"],
  "problem_solved": "concise description of problem being solved",
  "value_proposition": "key value delivered to customers",
  "technologies": ["tech1", "tech2", "tech3"],
  "data_sources": ["data source 1", "data source 2"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "industries": ["provider industry categories"],
  "target_customers": ["who uses this product/service"]
}}

Guidelines:
- Domain: The PROVIDER company's industry (what category of company they are)
- Technologies: Specific tech mentioned
- Data sources: Types of data used
- Keywords: Features and capabilities of the PROVIDER company
- Industries: Provider industry categories
- Target customers: Who BUYS/USES the product (these are NOT the companies we want to find)"""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a market research analyst expert at extracting structured information from company descriptions. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            # Generate search dimensions based on extracted components
            search_dimensions = self._generate_search_dimensions(result)
            
            return QueryComponents(
                domain=result.get('domain', []),
                problem_solved=result.get('problem_solved', ''),
                value_proposition=result.get('value_proposition', ''),
                technologies=result.get('technologies', []),
                data_sources=result.get('data_sources', []),
                search_dimensions=search_dimensions,
                keywords=result.get('keywords', []),
                industries=result.get('industries', [])
            )
            
        except Exception as e:
            logger.error(f"AI decomposition failed: {e}")
            return self._rule_based_decomposition(query)
    
    def _rule_based_decomposition(self, query: str) -> QueryComponents:
        """Fallback rule-based decomposition if AI is unavailable"""
        
        # Simple keyword extraction
        tech_keywords = ['AI', 'ML', 'machine learning', 'agent', 'automation', 'scraping', 'NLP', 'neural network']
        domain_keywords = ['hedge fund', 'trading', 'fintech', 'finance', 'investment']
        data_keywords = ['market data', 'pricing', 'news', 'financial data', 'real-time']
        
        query_lower = query.lower()
        
        technologies = [kw for kw in tech_keywords if kw.lower() in query_lower]
        domain = [kw for kw in domain_keywords if kw.lower() in query_lower]
        data_sources = [kw for kw in data_keywords if kw.lower() in query_lower]
        
        # Extract potential keywords (simple word extraction)
        words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        keywords = list(set(words))[:10]
        
        search_dimensions = self._generate_search_dimensions({
            'domain': domain,
            'technologies': technologies,
            'data_sources': data_sources,
            'keywords': keywords
        })
        
        return QueryComponents(
            domain=domain,
            problem_solved="Extracted from query",
            value_proposition="Extracted from query",
            technologies=technologies,
            data_sources=data_sources,
            search_dimensions=search_dimensions,
            keywords=keywords,
            industries=domain
        )
    
    def _generate_search_dimensions(self, components: Dict) -> Dict[str, List[str]]:
        """
        Generate targeted search queries for each market intelligence dimension
        """
        dimensions = {}
        
        domain = components.get('domain', [])
        technologies = components.get('technologies', [])
        keywords = components.get('keywords', [])
        
        # Dimension 1: Key Players & Companies (SOFTWARE/SERVICE PROVIDERS)
        dimensions['companies'] = []
        
        # Get target customers to search for companies serving them
        target_customers = components.get('target_customers', [])
        
        if domain:
            for d in domain[:2]:
                dimensions['companies'].extend([
                    f"software companies for {d}",
                    f"SaaS platforms for {d}",
                    f"startups building {d} tools"
                ])
        
        # If we know the target customers, search for companies serving them
        if target_customers:
            for customer in target_customers[:2]:
                dimensions['companies'].extend([
                    f"software companies serving {customer}",
                    f"AI platforms for {customer}",
                    f"{customer} technology vendors"
                ])
        
        if technologies and target_customers:
            dimensions['companies'].append(f"{technologies[0]} software for {target_customers[0]}")
        elif technologies and domain:
            dimensions['companies'].append(f"{technologies[0]} companies in {domain[0]}")
        
        # Dimension 2: Technology Landscape
        dimensions['technology'] = []
        if technologies:
            for tech in technologies[:3]:
                dimensions['technology'].extend([
                    f"{tech} in {domain[0] if domain else 'finance'}",
                    f"{tech} tools for {domain[0] if domain else 'trading'}"
                ])
        if domain:
            dimensions['technology'].append(f"technology stack for {domain[0]}")
        
        # Dimension 3: Business Models
        dimensions['business_models'] = []
        if domain:
            dimensions['business_models'].extend([
                f"{domain[0]} SaaS pricing models",
                f"how {domain[0]} companies monetize",
                f"{domain[0]} business models"
            ])
        
        # Dimension 4: Innovations & Trends
        dimensions['innovations'] = []
        if domain and technologies:
            dimensions['innovations'].extend([
                f"latest innovations in {domain[0]}",
                f"new {technologies[0]} solutions for {domain[0]}",
                f"emerging trends in {domain[0]}",
                f"{domain[0]} startup innovations 2024 2025"
            ])
        
        # Dimension 5: Market Analysis
        dimensions['market_trends'] = []
        if domain:
            dimensions['market_trends'].extend([
                f"{domain[0]} market trends 2024",
                f"{domain[0]} industry analysis",
                f"future of {domain[0]}"
            ])
        
        # Dimension 6: Competitive Intelligence
        dimensions['competitive'] = []
        if keywords:
            dimensions['competitive'].extend([
                f"competitors in {' '.join(keywords[:2])}",
                f"alternatives to {' '.join(keywords[:2])}"
            ])
        
        return dimensions
    
    def generate_search_plan(self, components: QueryComponents) -> Dict:
        """
        Generate a comprehensive search plan based on decomposed components
        
        Returns:
            Dictionary with search strategy and prioritized queries
        """
        plan = {
            'summary': {
                'domain': ', '.join(components.domain),
                'key_technologies': ', '.join(components.technologies[:5]),
                'search_scope': len(components.search_dimensions),
                'total_queries': sum(len(queries) for queries in components.search_dimensions.values())
            },
            'dimensions': {},
            'priority_order': [
                'companies',
                'competitive', 
                'technology',
                'innovations',
                'business_models',
                'market_trends'
            ]
        }
        
        # Organize search queries by dimension with descriptions
        dimension_descriptions = {
            'companies': 'Key players, established companies, and emerging startups in the space',
            'technology': 'Technology stack, tools, and platforms being used',
            'business_models': 'Pricing strategies, monetization approaches, and revenue models',
            'innovations': 'Recent innovations, new approaches, and differentiators',
            'market_trends': 'Market dynamics, growth trends, and future outlook',
            'competitive': 'Competitor identification and landscape mapping'
        }
        
        for dimension, queries in components.search_dimensions.items():
            plan['dimensions'][dimension] = {
                'description': dimension_descriptions.get(dimension, ''),
                'queries': queries,
                'priority': plan['priority_order'].index(dimension) if dimension in plan['priority_order'] else 99
            }
        
        return plan

