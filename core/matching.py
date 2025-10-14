# matching_service.py

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np

from storage import StorageManager
from config import openai_client, EMBEDDING_MODEL

logger = logging.getLogger("matching")
logging.basicConfig(level=logging.INFO)


class MatchingService:
    """
    Finds relevant articles for portfolio companies
    Strategies: Direct mentions, keywords, industry, semantic similarity
    """
    
    def __init__(self):
        self.storage = StorageManager()
        self.openai = openai_client
    
    def match_all_portfolio_companies(self, days_back: int = 7):
        """
        Find matches for all portfolio companies
        Only checks recent articles (efficiency)
        
        Args:
            days_back: Only check articles from last N days
        """
        logger.info(f"Starting matching for all portfolio companies (last {days_back} days)\n")
        
        # Get all portfolio companies
        companies = self._get_all_companies()
        logger.info(f"Found {len(companies)} portfolio companies")
        
        # Get recent articles
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_articles = self._get_recent_articles(cutoff_date)
        logger.info(f"Found {len(recent_articles)} recent articles\n")
        
        total_matches = 0
        
        for idx, company in enumerate(companies, 1):
            logger.info(f"[{idx}/{len(companies)}] Matching for: {company['company_name']}")
            
            matches = self.find_matches_for_company(company, recent_articles)
            
            if matches:
                # Store matches
                self._store_matches(matches)
                total_matches += len(matches)
                logger.info(f"  ✓ Found {len(matches)} relevant articles")
            else:
                logger.info(f"  ⊘ No matches found")
        
        logger.info(f"\n✓ Matching complete! Total matches: {total_matches}")
    
    def find_matches_for_company(
        self, 
        company: Dict,
        articles: List[Dict]
    ) -> List[Dict]:
        """
        Find all relevant articles for a single company
        
        Returns:
            List of match dictionaries
        """
        matches = []
        company_id = company['id']
        company_name = company['company_name']
        
        # Strategy 1: Direct company name mentions
        direct = self._find_direct_mentions(company_name, articles, company_id)
        matches.extend(direct)
        
        # Strategy 2: Keyword overlap
        if company.get('keywords'):
            keyword_matches = self._find_keyword_matches(
                company['keywords'], 
                articles, 
                company_id
            )
            matches.extend(keyword_matches)
        
        # Strategy 3: Industry matches
        if company.get('industry'):
            industry_matches = self._find_industry_matches(
                company['industry'],
                articles,
                company_id
            )
            matches.extend(industry_matches)
        
        # Strategy 4: Target customer matches
        if company.get('target_customer'):
            customer_matches = self._find_target_customer_matches(
                company['target_customer'],
                articles,
                company_id
            )
            matches.extend(customer_matches)
        
        # Strategy 5: Semantic similarity (best for finding similar products)
        semantic_matches = self._find_semantic_matches(company, articles, company_id)
        matches.extend(semantic_matches)
        
        # Deduplicate (same article matched multiple ways)
        unique_matches = self._deduplicate_matches(matches)
        
        return unique_matches
    
    def _find_direct_mentions(
        self,
        company_name: str,
        articles: List[Dict],
        company_id: str
    ) -> List[Dict]:
        """Find articles that directly mention the company name"""
        matches = []
        
        for article in articles:
            # Check in company_names array (from GPT extraction)
            if article.get('company_names') and company_name in article['company_names']:
                matches.append({
                    'portfolio_company_id': company_id,
                    'article_id': article['id'],
                    'match_type': 'direct_mention',
                    'relevance_score': 1.0,
                    'match_reason': f'Article mentions {company_name} directly'
                })
        
        return matches
    
    def _find_keyword_matches(
        self,
        keywords: List[str],
        articles: List[Dict],
        company_id: str
    ) -> List[Dict]:
        """Find articles containing company keywords"""
        matches = []
        
        # Convert keywords to lowercase for matching
        keywords_lower = [kw.lower().strip() for kw in keywords]
        
        for article in articles:
            content = (
                article.get('raw_content', '') + ' ' + 
                article.get('title', '') + ' ' + 
                article.get('summary', '')
            ).lower()
            
            # Count matching keywords
            matched_keywords = [kw for kw in keywords_lower if kw in content]
            
            if len(matched_keywords) >= 2:  # At least 2 keyword matches
                score = min(len(matched_keywords) * 0.15, 0.9)
                matches.append({
                    'portfolio_company_id': company_id,
                    'article_id': article['id'],
                    'match_type': 'keyword',
                    'relevance_score': score,
                    'match_reason': f'Contains keywords: {", ".join(matched_keywords[:3])}'
                })
        
        return matches
    
    def _find_industry_matches(
        self,
        industry: str,
        articles: List[Dict],
        company_id: str
    ) -> List[Dict]:
        """Find articles in the same industry"""
        matches = []
        
        industry_lower = industry.lower()
        
        for article in articles:
            content = (
                article.get('raw_content', '') + ' ' + 
                article.get('title', '') + ' ' + 
                article.get('summary', '')
            ).lower()
            
            if industry_lower in content:
                matches.append({
                    'portfolio_company_id': company_id,
                    'article_id': article['id'],
                    'match_type': 'industry_trend',
                    'relevance_score': 0.6,
                    'match_reason': f'Article about {industry}'
                })
        
        return matches
    
    def _find_target_customer_matches(
        self,
        target_customer: str,
        articles: List[Dict],
        company_id: str
    ) -> List[Dict]:
        """Find articles about the company's target customers"""
        matches = []
        
        target_lower = target_customer.lower()
        
        for article in articles:
            content = (
                article.get('raw_content', '') + ' ' + 
                article.get('title', '') + ' ' + 
                article.get('summary', '')
            ).lower()
            
            if target_lower in content:
                matches.append({
                    'portfolio_company_id': company_id,
                    'article_id': article['id'],
                    'match_type': 'target_customer',
                    'relevance_score': 0.7,
                    'match_reason': f'Article mentions target customer: {target_customer}'
                })
        
        return matches
    
    def _find_semantic_matches(
        self,
        company: Dict,
        articles: List[Dict],
        company_id: str
    ) -> List[Dict]:
        """Find articles about similar products using embeddings"""
        matches = []
        
        if not self.openai:
            return matches
        
        # Build query from company description
        query_text = f"{company.get('what_they_build', '')} {company.get('description', '')}"
        
        if not query_text.strip():
            return matches
        
        try:
            # Generate embedding
            response = self.openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query_text[:1000]  # Limit length
            )
            query_embedding = response.data[0].embedding
            
            # Search for similar chunks
            similar = self.storage.search_similar_chunks(
                query_embedding=query_embedding,
                threshold=0.75,
                limit=15
            )
            
            # Convert to matches
            seen_articles = set()
            for item in similar:
                article_id = item['article_id']
                if article_id not in seen_articles:
                    matches.append({
                        'portfolio_company_id': company_id,
                        'article_id': article_id,
                        'match_type': 'similar_model',
                        'relevance_score': item['similarity'],
                        'match_reason': f'Similar business model (similarity: {item["similarity"]:.2f})'
                    })
                    seen_articles.add(article_id)
        
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
        
        return matches
    
    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Remove duplicate matches (same article matched multiple ways)
        Keep the match with highest score
        """
        by_article = {}
        
        for match in matches:
            article_id = match['article_id']
            
            if article_id not in by_article:
                by_article[article_id] = match
            else:
                # Keep match with higher score
                if match['relevance_score'] > by_article[article_id]['relevance_score']:
                    by_article[article_id] = match
        
        return list(by_article.values())
    
    def _get_all_companies(self) -> List[Dict]:
        """Get all portfolio companies"""
        response = self.storage.supabase.table('portfolio_companies').select('*').execute()
        return response.data or []
    
    def _get_recent_articles(self, since_date: datetime) -> List[Dict]:
        """Get articles published after a certain date"""
        response = self.storage.supabase.table('articles').select(
            'id, title, url, raw_content, summary, company_names, published_date'
        ).gte('published_date', since_date.isoformat()).execute()
        
        return response.data or []
    
    def _store_matches(self, matches: List[Dict]):
        """Store article matches to database"""
        if not matches:
            return
        
        try:
            # Upsert to handle duplicates
            self.storage.supabase.table('article_matches').upsert(matches).execute()
        except Exception as e:
            logger.error(f"Failed to store matches: {e}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Match portfolio companies to relevant articles')
    parser.add_argument('--days', type=int, default=7, help='Check articles from last N days (default: 7)')
    
    args = parser.parse_args()
    
    matcher = MatchingService()
    matcher.match_all_portfolio_companies(days_back=args.days)