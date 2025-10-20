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
        
        prompt = f"""Analyze this company/market description and extract structured information for market discovery:

Query: {query}

CRITICAL INSTRUCTIONS:
1. Identify the SPECIFIC USE CASE and APPLICATION mentioned in the query
2. Distinguish between PROVIDER and CUSTOMER industries
3. Extract ALL key context: what problem is solved, what data is used, who benefits

IMPORTANT:
- If query says "company that does X for Y" → Y is the customer, X is what the company does
- Domain should describe the PROVIDER company's industry/use case (e.g., "hedge fund analytics", "trading analysis", "market data intelligence")
- Technologies: Specific tech AND data types mentioned (e.g., "AI agents", "market data scraping", "financial news analysis")
- Keywords: MUST include the problem being solved, data sources, and customer type
- Target customers: Who BUYS/USES the product

Extract and return ONLY valid JSON with this EXACT structure:
{{
  "domain": ["specific provider industry with use case context"],
  "technologies": ["tech1", "tech2", "data source 1", "data source 2"],
  "keywords": ["problem/use case", "data type 1", "data type 2", "customer type", "key capability 1", "key capability 2"],
  "target_customers": ["who uses this product/service"],
  "core_use_case": "one sentence describing what these companies do"
}}

EXAMPLES:
Query: "company that does hedge fund analysis using AI agents that scrape market data"
→ domain: ["hedge fund analytics", "AI-powered trading analysis"]
→ technologies: ["AI", "AI agents", "web scraping", "market data analysis"]
→ keywords: ["hedge fund analysis", "market data", "trading analysis", "financial data", "automated updates", "hedge funds"]
→ target_customers: ["hedge funds", "trading firms"]
→ core_use_case: "AI-powered market data analysis and automated insights for hedge funds and trading firms"

Focus on discovering companies, technologies, and business models that match this SPECIFIC use case."""

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
                problem_solved="",  # Not needed for market discovery
                value_proposition="",  # Not needed for market discovery
                technologies=result.get('technologies', []),
                data_sources=[],  # Not needed for market discovery
                search_dimensions=search_dimensions,
                keywords=result.get('keywords', []),
                industries=result.get('domain', [])  # Use domain as industries
            )
            
        except Exception as e:
            logger.error(f"AI decomposition failed: {e}")
            return self._rule_based_decomposition(query)
    
    def _rule_based_decomposition(self, query: str) -> QueryComponents:
        """Fallback rule-based decomposition if AI is unavailable"""

        # Comprehensive technology keywords list
        tech_keywords = [
            # AI/ML & Intelligence
            'AI', 'ML', 'machine learning', 'artificial intelligence', 'deep learning', 'neural network', 
            'NLP', 'natural language processing', 'computer vision', 'predictive analytics', 'automation',
            'robotic process automation', 'RPA', 'intelligent automation', 'cognitive computing',
            'machine vision', 'speech recognition', 'text analysis', 'sentiment analysis',
            
            # Data & Analytics
            'data analysis', 'data science', 'big data', 'data mining', 'business intelligence', 'BI',
            'data visualization', 'dashboard', 'reporting', 'analytics', 'statistical analysis',
            'data engineering', 'ETL', 'data pipeline', 'data warehouse', 'data lake',
            'data modeling', 'data governance', 'data quality', 'data profiling', 'data catalog',
            'real-time analytics', 'streaming analytics', 'predictive modeling', 'forecasting',
            
            # Cloud & Infrastructure
            'cloud', 'SaaS', 'PaaS', 'IaaS', 'AWS', 'Azure', 'GCP', 'Google Cloud', 'Microsoft Azure',
            'cloud computing', 'serverless', 'microservices', 'containerization', 'Docker', 'Kubernetes',
            'edge computing', 'distributed computing', 'scalability', 'infrastructure',
            'multi-cloud', 'hybrid cloud', 'private cloud', 'public cloud', 'cloud migration',
            'load balancing', 'auto-scaling', 'CDN', 'content delivery network',
            
            # Web & Mobile Development
            'web', 'mobile', 'iOS', 'Android', 'React', 'Vue', 'Angular', 'Node.js', 'JavaScript',
            'TypeScript', 'Python', 'Java', 'C#', 'Go', 'Rust', 'PHP', 'Ruby', 'Swift', 'Kotlin',
            'responsive design', 'progressive web app', 'PWA', 'cross-platform', 'Flutter', 'React Native',
            'Xamarin', 'Ionic', 'Cordova', 'HTML5', 'CSS3', 'Bootstrap', 'jQuery',
            
            # APIs & Integration
            'API', 'REST', 'GraphQL', 'webhook', 'integration', 'middleware', 'microservices',
            'service-oriented architecture', 'SOA', 'event-driven', 'message queue', 'Kafka',
            'API gateway', 'API management', 'API documentation', 'OpenAPI', 'Swagger',
            'web services', 'SOAP', 'gRPC', 'message broker', 'event streaming',
            
            # Blockchain & Crypto
            'blockchain', 'crypto', 'cryptocurrency', 'Bitcoin', 'Ethereum', 'DeFi', 'NFT',
            'smart contract', 'distributed ledger', 'Web3', 'metaverse', 'virtual reality', 'VR',
            'augmented reality', 'AR', 'mixed reality', 'MR', 'crypto trading', 'digital assets',
            'tokenization', 'consensus mechanism', 'proof of work', 'proof of stake',
            
            # Security & Privacy
            'cybersecurity', 'security', 'encryption', 'privacy', 'GDPR', 'compliance', 'authentication',
            'authorization', 'identity management', 'zero trust', 'penetration testing', 'vulnerability',
            'threat detection', 'incident response', 'security monitoring', 'access control',
            'data protection', 'privacy by design', 'security audit', 'risk assessment',
            
            # IoT & Hardware
            'IoT', 'internet of things', 'sensors', 'embedded systems', 'hardware', 'firmware',
            'edge devices', 'smart devices', 'wearables', 'industrial IoT', 'IIoT',
            'sensor networks', 'device management', 'fleet management', 'asset tracking',
            'smart home', 'smart city', 'connected devices', 'M2M', 'machine to machine',
            
            # Emerging Technologies
            'quantum computing', 'quantum', '5G', '6G', 'autonomous vehicles', 'self-driving',
            'robotics', 'drones', 'UAV', 'satellite', 'space tech', 'biotech', 'fintech',
            'healthtech', 'edtech', 'cleantech', 'greentech', 'agtech', 'proptech',
            'nanotechnology', 'synthetic biology', 'gene editing', 'CRISPR',
            
            # Business & Process
            'workflow', 'process', 'optimization', 'efficiency', 'productivity', 'collaboration',
            'project management', 'CRM', 'ERP', 'HR', 'accounting', 'finance', 'marketing',
            'sales', 'customer service', 'support', 'ticketing', 'helpdesk', 'workflow automation',
            'business process management', 'BPM', 'process mining', 'digital transformation',
            
            # Communication & Collaboration
            'communication', 'messaging', 'chat', 'video conferencing', 'telephony', 'VoIP',
            'collaboration', 'teamwork', 'remote work', 'virtual office', 'digital workspace',
            'unified communications', 'UC', 'team collaboration', 'document sharing',
            'screen sharing', 'web conferencing', 'instant messaging', 'presence',
            
            # E-commerce & Financial
            'e-commerce', 'ecommerce', 'marketplace', 'online store', 'payment', 'fintech',
            'digital wallet', 'cryptocurrency', 'trading', 'investment', 'wealth management',
            'banking', 'financial services', 'payment processing', 'POS', 'point of sale',
            'subscription billing', 'recurring billing', 'payment gateway', 'PCI compliance',
            
            # Content & Media
            'content management', 'CMS', 'digital marketing', 'SEO', 'social media', 'streaming',
            'video', 'audio', 'podcast', 'blog', 'publishing', 'media', 'entertainment',
            'content creation', 'video editing', 'audio editing', 'live streaming',
            'content delivery', 'media management', 'digital asset management', 'DAM',
            
            # Automation & RPA
            'automation', 'RPA', 'robotic process automation', 'workflow automation',
            'process automation', 'intelligent automation', 'business process automation', 'BPA',
            'task automation', 'script automation', 'test automation', 'deployment automation',
            'infrastructure automation', 'configuration management', 'orchestration',
            
            # Database & Storage
            'database', 'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Elasticsearch',
            'data storage', 'file storage', 'object storage', 'backup', 'recovery',
            'data replication', 'data synchronization', 'data archiving', 'data retention',
            'database optimization', 'query optimization', 'indexing', 'partitioning',
            
            # Development & DevOps
            'DevOps', 'CI/CD', 'continuous integration', 'continuous deployment', 'version control',
            'Git', 'GitHub', 'GitLab', 'testing', 'QA', 'quality assurance', 'deployment',
            'monitoring', 'logging', 'APM', 'application performance monitoring',
            'infrastructure as code', 'IaC', 'configuration management', 'container orchestration',
            
            # Additional Tech Terms
            'machine learning', 'deep learning', 'neural networks', 'artificial intelligence',
            'data science', 'business intelligence', 'cloud computing', 'serverless computing',
            'microservices architecture', 'API-first', 'headless', 'composable',
            'low-code', 'no-code', 'citizen development', 'visual programming',
            'machine learning ops', 'MLOps', 'data ops', 'DataOps', 'model ops', 'ModelOps'
        ]

        query_lower = query.lower()
        technologies = [kw for kw in tech_keywords if kw.lower() in query_lower]

        # Extract domain and customer info from the query itself
        domain = []
        target_customers = []
        
        # Extract potential keywords (simple word extraction)
        words = re.findall(r'\b[a-z]{4,}\b', query_lower)
        keywords = list(set(words))[:10]

        search_dimensions = self._generate_search_dimensions({
            'domain': domain,
            'technologies': technologies,
            'keywords': keywords,
            'target_customers': target_customers
        })
        
        return QueryComponents(
            domain=domain,
            problem_solved="",  # Not needed for market discovery
            value_proposition="",  # Not needed for market discovery
            technologies=technologies,
            data_sources=[],  # Not needed for market discovery
            search_dimensions=search_dimensions,
            keywords=keywords,
            industries=domain
        )
    
    def _generate_search_dimensions(self, components: Dict) -> Dict[str, List[str]]:
        """
        Generate targeted search queries for market discovery across multiple dimensions
        Focus on discovering NEW companies, technologies, and business models
        """
        dimensions = {}

        domain = components.get('domain', [])
        technologies = components.get('technologies', [])
        keywords = components.get('keywords', [])
        target_customers = components.get('target_customers', [])
        core_use_case = components.get('core_use_case', '')

        # Build context-rich query base combining domain, technologies, and keywords
        # This ensures search results are highly relevant to the specific use case
        context_parts = []
        if domain:
            context_parts.extend(domain[:2])
        if technologies:
            context_parts.extend(technologies[:2])
        if keywords:
            # Prioritize keywords that describe the problem/use case
            context_parts.extend([kw for kw in keywords[:3] if len(kw) > 4])

        context_base = ' '.join(context_parts[:4])  # Limit to avoid too long queries

        # Dimension 1: Emerging Companies & Startups
        dimensions['companies'] = []

        # Use FULL context to ensure relevance
        if domain and keywords:
            # Combine domain + key capabilities for precision
            for d in domain[:2]:
                for kw in keywords[:2]:
                    dimensions['companies'].append(f"companies {d} {kw}")
                    dimensions['companies'].append(f"startups {kw} for {target_customers[0] if target_customers else d}")

        if context_base:
            dimensions['companies'].extend([
                f"new companies {context_base}",
                f"startups {context_base}",
                f"emerging {context_base} platforms"
            ])

        if technologies and target_customers:
            for tech in technologies[:2]:
                for customer in target_customers[:1]:
                    dimensions['companies'].append(f"{tech} platforms for {customer}")
        
        # Dimension 2: Technology Innovations
        dimensions['technology'] = []

        # Use context-rich queries combining tech + domain + use case
        if technologies and domain:
            for tech in technologies[:2]:
                for d in domain[:1]:
                    dimensions['technology'].extend([
                        f"{tech} for {d}",
                        f"{tech} {d} solutions",
                        f"{d} using {tech}"
                    ])

        if context_base:
            dimensions['technology'].extend([
                f"technologies for {context_base}",
                f"{context_base} technology stack"
            ])
        
        # Dimension 3: Business Model Innovation
        dimensions['business_models'] = []

        if context_base:
            dimensions['business_models'].extend([
                f"business models {context_base}",
                f"pricing models {context_base}",
                f"monetization {context_base}"
            ])

        if domain and target_customers:
            dimensions['business_models'].extend([
                f"business models {domain[0]} for {target_customers[0]}",
                f"SaaS pricing {domain[0]}"
            ])
        
        # Dimension 4: Market Innovations & Trends
        dimensions['innovations'] = []

        if context_base:
            dimensions['innovations'].extend([
                f"innovations in {context_base}",
                f"new approaches {context_base}",
                f"breakthrough {context_base}"
            ])

        if domain and technologies:
            dimensions['innovations'].extend([
                f"{technologies[0]} innovations in {domain[0]}",
                f"novel {domain[0]} solutions"
            ])
        
        # Dimension 5: Market Trends & Opportunities
        dimensions['market_trends'] = []

        if context_base:
            dimensions['market_trends'].extend([
                f"trends in {context_base}",
                f"market opportunities {context_base}",
                f"future of {context_base}"
            ])

        if target_customers and domain:
            dimensions['market_trends'].extend([
                f"trends in {domain[0]} for {target_customers[0]}",
                f"{target_customers[0]} technology trends"
            ])
        
        # Dimension 6: Competitive Landscape Discovery
        dimensions['competitive'] = []

        if context_base:
            dimensions['competitive'].extend([
                f"competitors in {context_base}",
                f"alternative {context_base} platforms",
                f"companies competing in {context_base}"
            ])

        if domain and target_customers:
            dimensions['competitive'].extend([
                f"vendors {domain[0]} for {target_customers[0]}",
                f"{domain[0]} service providers"
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

