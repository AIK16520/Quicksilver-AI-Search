# parsers/parser_factory.py

from typing import Dict
# from parsers.rss_parser import RSSParser
from parsers.beehive import BeehiveScraper
from parsers.productHunt import ProductHuntScraper


class ParserFactory:
    """Factory for creating parser instances based on source type"""
    
    @staticmethod
    def create(source_type: str, newsletter_id: str, url: str, config: Dict):
        """
        Create appropriate parser based on source type.
        
        Args:
            source_type: Type of source ('rss', 'beehive', 'web', 'email')
            newsletter_id: Newsletter UUID
            url: Source URL
            config: Parser-specific configuration (e.g., selectors for web scrapers)
        
        Returns:
            Parser instance that implements BaseParser interface
        
        Raises:
            ValueError: If source_type is unknown or not implemented
        
        Examples:
            >>> parser = ParserFactory.create('rss', 'id-123', 'https://feed.com/rss', {})
            >>> parser = ParserFactory.create('beehive', 'id-456', 'https://news.beehiiv.com', {})
        """
        source_type = source_type.lower().strip()
        
        # if source_type == 'rss':
        #     return RSSParser(newsletter_id, url, config)
        
        if source_type == 'beehive':
            return BeehiveScraper(newsletter_id, url, config)
        
        elif source_type == 'producthunt':
            return ProductHuntScraper(newsletter_id, url, config)
        
        # Future parser types:
        # elif source_type == 'web':
        #     return WebParser(newsletter_id, url, config)
        # 
        # elif source_type == 'email':
        #     return EmailParser(newsletter_id, url, config)
        # 
        # elif source_type == 'substack':
        #     return SubstackParser(newsletter_id, url, config)
        
        else:
            raise ValueError(
                f"Unknown source type: '{source_type}'. "
                f"Supported types: 'beehive', 'producthunt'"
            )
    
    @staticmethod
    def get_supported_types() -> list:
        """
        Get list of supported parser types.
        
        Returns:
            List of supported source type strings
        """
        return ['beehive', 'producthunt']