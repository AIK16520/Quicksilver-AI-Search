# competitor_discovery.py

import requests
import json
import time
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import quote
import re

from openai import OpenAI
from core.config import supabase_client, OPENAI_API_KEY, BRAVE_API_KEY, GPT_MODEL
from .competitor_config import CompetitorDiscoveryConfig

logger = logging.getLogger("competitor_discovery")

@dataclass
class DiscoveredCompetitor:
    """Represents a newly discovered competitor"""
    name: str
    industry: str
    keywords: List[str]
    description: str
    confidence_score: float
    source_url: str
    search_signal: str  # The keyword/industry that led to discovery

class CompetitorDiscoveryService:
    """
    Discovers new competitors using keywords, industries, and web search
    Goes beyond the existing portfolio map to find emerging competitors
    """

    def __init__(self, config: CompetitorDiscoveryConfig = None):
        self.supabase = supabase_client
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.brave_key = BRAVE_API_KEY
        self.config = config or CompetitorDiscoveryConfig()

        # Known competitor patterns to avoid false positives
        self.competitor_indicators = [
            'competitor', 'competitors', 'alternative to', 'similar to',
            'rivals', 'competitors include', 'top competitors',
            'leading competitors', 'key players in'
        ]

    def discover_competitors(
        self,
        keywords: List[str],
        industries: List[str],
        existing_competitors: Set[str],
        config: CompetitorDiscoveryConfig = None
    ) -> List[DiscoveredCompetitor]:
        """
        Discover new competitors using multiple strategies

        Args:
            keywords: List of keywords to search for
            industries: List of industries to search in
            existing_competitors: Set of already known competitor names
            config: Configuration for discovery behavior (uses instance config if None)
        """
        if config:
            self.config = config

        discovered = []

        logger.info(f"Starting competitor discovery for {len(keywords)} keywords and {len(industries)} industries")

        # Strategy 1: Search for competitors using keywords
        if self.config.include_web_search:
            keyword_competitors = self._search_keyword_competitors(keywords, existing_competitors, self.config.max_results // 2)
            discovered.extend(keyword_competitors)

        # Strategy 2: Search for competitors in specific industries
        if self.config.include_web_search:
            industry_competitors = self._search_industry_competitors(industries, existing_competitors, self.config.max_results // 2)
            discovered.extend(industry_competitors)

        # Strategy 3: Use AI to find related companies
        if self.config.include_ai_powered and self.openai_client:
            ai_competitors = self._search_ai_powered_competitors(keywords, industries, existing_competitors, self.config.max_results // 3)
            discovered.extend(ai_competitors)

        # Filter and deduplicate
        filtered_competitors = self._filter_competitors(discovered, existing_competitors)
        unique_competitors = self._deduplicate_and_rank(filtered_competitors, existing_competitors)

        logger.info(f"Discovered {len(unique_competitors)} new potential competitors after filtering")
        return unique_competitors[:self.config.max_results]

    def _filter_competitors(
        self,
        competitors: List[DiscoveredCompetitor],
        existing_competitors: Set[str]
    ) -> List[DiscoveredCompetitor]:
        """Filter competitors based on configuration criteria"""
        filtered = []

        for comp in competitors:
            # Skip if already in existing competitors
            if comp.name.lower() in existing_competitors:
                continue

            # Skip if confidence is too low
            if comp.confidence_score < self.config.min_confidence_score:
                continue

            # Skip if industry is in exclude list
            if comp.industry in self.config.exclude_industries:
                continue

            # Skip if company name matches exclude patterns
            should_exclude = False
            for pattern in self.config.exclude_company_patterns:
                if pattern.lower() in comp.name.lower():
                    should_exclude = True
                    break

            if should_exclude:
                continue

            filtered.append(comp)

        return filtered

    def _search_keyword_competitors(
        self,
        keywords: List[str],
        existing_competitors: Set[str],
        max_results: int
    ) -> List[DiscoveredCompetitor]:
        """Search for competitors using specific keywords"""
        discovered = []

        # Generate search queries from configuration templates
        keyword_queries = []
        for template in self.config.keyword_search_queries:
            # Replace {keywords} placeholder with actual keywords
            query = template.replace('{keywords}', ' '.join(keywords[:3]))
            keyword_queries.append(query)

        for query in keyword_queries:
            try:
                search_results = self._brave_search(query, num_results=8)

                for result in search_results[:3]:  # Process top results
                    potential_competitors = self._extract_competitors_from_result(result, keywords[0])
                    for comp in potential_competitors:
                        if comp.name not in existing_competitors:
                            discovered.append(comp)

            except Exception as e:
                logger.error(f"Error searching keywords: {e}")
                continue

        return discovered[:max_results]

    def _search_industry_competitors(
        self,
        industries: List[str],
        existing_competitors: Set[str],
        max_results: int
    ) -> List[DiscoveredCompetitor]:
        """Search for competitors in specific industries"""
        discovered = []

        for industry in industries[:3]:  # Limit to top 3 industries
            queries = []
            for template in self.config.industry_search_queries:
                # Replace {industry} placeholder with actual industry
                query = template.replace('{industry}', industry)
                queries.append(query)

            for query in queries:
                try:
                    search_results = self._brave_search(query, num_results=8)

                    for result in search_results[:2]:  # Process top results
                        potential_competitors = self._extract_competitors_from_result(result, industry)
                        for comp in potential_competitors:
                            if comp.name not in existing_competitors:
                                discovered.append(comp)

                except Exception as e:
                    logger.error(f"Error searching industry {industry}: {e}")
                    continue

        return discovered[:max_results]

    def _search_ai_powered_competitors(
        self,
        keywords: List[str],
        industries: List[str],
        existing_competitors: Set[str],
        max_results: int
    ) -> List[DiscoveredCompetitor]:
        """Use AI to find related companies and potential competitors"""
        if not self.openai_client:
            return []

        discovered = []

        # Create a comprehensive prompt for AI
        context = f"""
        Keywords: {', '.join(keywords[:5])}
        Industries: {', '.join(industries[:3])}

        Based on these keywords and industries, suggest 5-7 companies that could be competitors or similar players.
        Focus on companies that might not be widely known but are relevant.
        Return only company names, one per line.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a market research expert. Suggest relevant companies based on keywords and industries."},
                    {"role": "user", "content": context}
                ],
                max_tokens=200,
                temperature=0.7
            )

            ai_suggestions = response.choices[0].message.content.strip().split('\n')
            ai_company_names = [name.strip('- ').strip() for name in ai_suggestions if name.strip()]

            # Research each suggested company
            for company_name in ai_company_names[:5]:
                if company_name and company_name not in existing_competitors:
                    competitor = self._research_company(company_name, keywords[0])
                    if competitor:
                        discovered.append(competitor)

        except Exception as e:
            logger.error(f"Error in AI-powered competitor search: {e}")

        return discovered[:max_results]

    def _brave_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform Brave search for a query"""
        if not self.brave_key:
            return []

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

            return data.get("web", {}).get("results", [])

        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

    def _extract_competitors_from_result(self, result: Dict, signal: str) -> List[DiscoveredCompetitor]:
        """Extract potential competitor names from search result"""
        competitors = []
        title = result.get('title', '').lower()
        description = result.get('description', '').lower()
        url = result.get('url', '')

        # Look for competitor indicators in title and description
        text_to_analyze = f"{title} {description}"

        # Simple pattern matching for competitor extraction
        # This is a basic implementation - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text_to_analyze)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in self.competitor_indicators):
                # Extract company names (basic pattern: capitalized words)
                words = sentence.split()
                for i, word in enumerate(words):
                    if (word.istitle() and len(word) > 2 and
                        i > 0 and words[i-1].lower() not in ['the', 'a', 'an']):

                        potential_name = word
                        # Add next word if it seems part of company name
                        if i + 1 < len(words) and words[i+1].istitle():
                            potential_name += f" {words[i+1]}"

                        if len(potential_name) > 2:
                            competitors.append(DiscoveredCompetitor(
                                name=potential_name.title(),
                                industry=signal,
                                keywords=[signal],
                                description=sentence.strip(),
                                confidence_score=0.6,  # Basic confidence
                                source_url=url,
                                search_signal=signal
                            ))

        return competitors[:3]  # Limit per result

    def _research_company(self, company_name: str, signal: str) -> Optional[DiscoveredCompetitor]:
        """Research a specific company to gather more information"""
        try:
            # Quick search for the company
            search_results = self._brave_search(f'"{company_name}" company', num_results=3)

            if search_results:
                result = search_results[0]
                return DiscoveredCompetitor(
                    name=company_name,
                    industry=signal,
                    keywords=[signal],
                    description=result.get('description', ''),
                    confidence_score=0.8,  # Higher confidence for AI suggestions
                    source_url=result.get('url', ''),
                    search_signal=signal
                )

        except Exception as e:
            logger.error(f"Error researching company {company_name}: {e}")

        return None

    def _deduplicate_and_rank(
        self,
        competitors: List[DiscoveredCompetitor],
        existing_competitors: Set[str]
    ) -> List[DiscoveredCompetitor]:
        """Remove duplicates and rank by confidence"""
        seen_names = set()
        unique_competitors = []

        for comp in competitors:
            if comp.name.lower() not in seen_names and comp.name not in existing_competitors:
                seen_names.add(comp.name.lower())
                unique_competitors.append(comp)

        # Sort by confidence score (highest first)
        unique_competitors.sort(key=lambda x: x.confidence_score, reverse=True)

        return unique_competitors
