# competitor_config.py

"""
Configuration options for competitor discovery
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class CompetitorDiscoveryConfig:
    """
    Configuration for competitor discovery behavior
    """
    # Search behavior
    max_results: int = 10
    search_depth: str = "moderate"  # "light", "moderate", "thorough"
    include_ai_powered: bool = True
    include_web_search: bool = True

    # Filtering criteria
    min_confidence_score: float = 0.5
    exclude_industries: List[str] = None
    exclude_company_patterns: List[str] = None

    # Search query customization
    keyword_search_queries: List[str] = None
    industry_search_queries: List[str] = None

    # Rate limiting
    requests_per_minute: int = 30
    max_concurrent_searches: int = 3

    def __post_init__(self):
        if self.exclude_industries is None:
            self.exclude_industries = []
        if self.exclude_company_patterns is None:
            self.exclude_company_patterns = []
        if self.keyword_search_queries is None:
            self.keyword_search_queries = [
                "competitors in {keywords}",
                "companies similar to {keywords}",
                "alternatives to {keywords}",
                "{keywords} competitors list",
                "top {keywords} companies"
            ]
        if self.industry_search_queries is None:
            self.industry_search_queries = [
                "top companies in {industry}",
                "leading {industry} companies",
                "key players in {industry} sector",
                "best {industry} companies",
                "{industry} market leaders"
            ]

# Default configurations for different use cases
LIGHT_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=5,
    search_depth="light",
    include_ai_powered=False,
    include_web_search=True,
    min_confidence_score=0.7,
    requests_per_minute=15
)

MODERATE_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=10,
    search_depth="moderate",
    include_ai_powered=True,
    include_web_search=True,
    min_confidence_score=0.5,
    requests_per_minute=30
)

THOROUGH_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=20,
    search_depth="thorough",
    include_ai_powered=True,
    include_web_search=True,
    min_confidence_score=0.3,
    requests_per_minute=60,
    max_concurrent_searches=5
)

# Industry-specific configurations
FINTECH_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=15,
    search_depth="moderate",
    include_ai_powered=True,
    min_confidence_score=0.4,
    exclude_industries=["Banking", "Insurance"],  # Avoid traditional finance
    requests_per_minute=45
)

SAAS_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=12,
    search_depth="moderate",
    include_ai_powered=True,
    min_confidence_score=0.5,
    exclude_company_patterns=["Microsoft", "Google", "Amazon"],  # Focus on pure-play SaaS
    requests_per_minute=35
)

AI_DISCOVERY = CompetitorDiscoveryConfig(
    max_results=15,
    search_depth="thorough",
    include_ai_powered=True,
    min_confidence_score=0.4,
    requests_per_minute=50
)
