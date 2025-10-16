# report_formatter.py

"""
Format market intelligence reports for display and API consumption
"""

from typing import Dict, Any
from .market_intelligence import MarketIntelligenceReport, DimensionResults
from .entity_extraction import EntityExtractionService


class ReportFormatter:
    """Formats market intelligence reports into structured, consumable formats"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractionService()
    
    def format_for_api(self, report: MarketIntelligenceReport, extract_entities: bool = True) -> Dict[str, Any]:
        """
        Format report for API consumption (JSON-serializable)
        """
        formatted_report = {
            'query': report.query,
            'generated_at': report.generated_at,
            
            # Query analysis
            'analysis': {
                'domain': report.components.domain,
                'problem_solved': report.components.problem_solved,
                'value_proposition': report.components.value_proposition,
                'technologies': report.components.technologies,
                'data_sources': report.components.data_sources,
                'keywords': report.components.keywords,
                'industries': report.components.industries
            },
            
            # Executive insights
            'executive_summary': report.executive_summary,
            'key_players': report.key_players,
            'emerging_trends': report.emerging_trends,
            'recommended_next_steps': report.recommended_next_steps,
            
            # Dimensional breakdowns
            'dimensions': {
                'companies': self._format_dimension(report.companies_landscape),
                'technology': self._format_dimension(report.technology_landscape),
                'business_models': self._format_dimension(report.business_models),
                'innovations': self._format_dimension(report.innovations),
                'market_trends': self._format_dimension(report.market_trends),
                'competitive': self._format_dimension(report.competitive_analysis)
            },
            
            # Metadata
            'metadata': {
                'total_articles': sum([
                    len(report.companies_landscape.articles),
                    len(report.technology_landscape.articles),
                    len(report.business_models.articles),
                    len(report.innovations.articles),
                    len(report.market_trends.articles),
                    len(report.competitive_analysis.articles)
                ]),
                'dimensions_analyzed': 6,
                'companies_identified': len(report.competitive_analysis.companies_mentioned)
            }
        }
        
        # Extract interactive entities
        if extract_entities:
            entities = self.entity_extractor.extract_entities(formatted_report, report.query)
            formatted_report['interactive_elements'] = self._format_entities(entities)
        
        return formatted_report
    
    def _format_entities(self, entities: Dict) -> Dict[str, Any]:
        """Format extracted entities for API response"""
        return {
            'companies': [
                {
                    'name': entity.name,
                    'context': entity.context,
                    'mentioned_in': entity.mentioned_in,
                    'actions': entity.actions,
                    'confidence': entity.confidence
                }
                for entity in entities.get('companies', [])[:10]  # Top 10
            ],
            'partnerships': [
                {
                    'description': entity.name,
                    'context': entity.context,
                    'actions': entity.actions,
                    'vague': entity.vague
                }
                for entity in entities.get('partnerships', [])[:5]
            ],
            'technologies': [
                {
                    'name': entity.name,
                    'context': entity.context,
                    'actions': entity.actions
                }
                for entity in entities.get('technologies', [])[:8]
            ],
            'vague_mentions': [
                {
                    'text': entity.name,
                    'resolution_query': entity.resolution_query,
                    'actions': entity.actions
                }
                for entity in entities.get('vague_mentions', [])
            ]
        }
    
    def _format_dimension(self, dimension: DimensionResults) -> Dict[str, Any]:
        """Format a single dimension result"""
        return {
            'name': dimension.dimension,
            'description': dimension.description,
            'key_findings': dimension.key_findings,
            'companies_mentioned': dimension.companies_mentioned,
            'technologies_mentioned': dimension.technologies_mentioned,
            'articles': [
                {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'description': article.get('description', '')[:200],  # Truncate
                    'source_query': article.get('query', '')
                }
                for article in dimension.articles
            ],
            'article_count': len(dimension.articles),
            'queries_used': dimension.queries_performed
        }
    
    def format_for_display(self, report: MarketIntelligenceReport) -> str:
        """
        Format report for terminal/console display
        """
        output = []
        
        # Header
        output.append("=" * 100)
        output.append("MARKET INTELLIGENCE REPORT")
        output.append("=" * 100)
        output.append(f"Query: {report.query}")
        output.append(f"Generated: {report.generated_at}")
        output.append("")
        
        # Executive Summary
        if report.executive_summary:
            output.append("━" * 100)
            output.append("EXECUTIVE SUMMARY")
            output.append("━" * 100)
            output.append(report.executive_summary)
            output.append("")
        
        # Key Players
        if report.key_players:
            output.append("━" * 100)
            output.append("KEY PLAYERS")
            output.append("━" * 100)
            for i, player in enumerate(report.key_players, 1):
                name = player.get('name', 'Unknown')
                desc = player.get('description', '')
                output.append(f"{i}. {name}")
                if desc:
                    output.append(f"   {desc}")
            output.append("")
        
        # Emerging Trends
        if report.emerging_trends:
            output.append("━" * 100)
            output.append("EMERGING TRENDS")
            output.append("━" * 100)
            for i, trend in enumerate(report.emerging_trends, 1):
                output.append(f"{i}. {trend}")
            output.append("")
        
        # Dimensional Analyses
        dimensions = [
            ('COMPANY LANDSCAPE', report.companies_landscape),
            ('TECHNOLOGY LANDSCAPE', report.technology_landscape),
            ('BUSINESS MODELS', report.business_models),
            ('INNOVATIONS & DIFFERENTIATORS', report.innovations),
            ('MARKET TRENDS', report.market_trends),
            ('COMPETITIVE ANALYSIS', report.competitive_analysis)
        ]
        
        for title, dimension in dimensions:
            output.append("━" * 100)
            output.append(title)
            output.append("━" * 100)
            
            if dimension.description:
                output.append(f"Focus: {dimension.description}")
                output.append("")
            
            # Key Findings
            if dimension.key_findings:
                output.append("Key Findings:")
                for finding in dimension.key_findings:
                    output.append(f"  • {finding}")
                output.append("")
            
            # Companies/Tech Mentioned
            if dimension.companies_mentioned:
                companies_str = ', '.join(dimension.companies_mentioned[:8])
                if len(dimension.companies_mentioned) > 8:
                    companies_str += f" (+{len(dimension.companies_mentioned) - 8} more)"
                output.append(f"Companies Mentioned: {companies_str}")
                output.append("")
            
            # Top Articles
            if dimension.articles:
                output.append(f"Top Articles ({len(dimension.articles)}):")
                for i, article in enumerate(dimension.articles[:5], 1):
                    output.append(f"  {i}. {article.get('title', 'Untitled')}")
                    output.append(f"     {article.get('url', '')}")
                if len(dimension.articles) > 5:
                    output.append(f"     ... and {len(dimension.articles) - 5} more articles")
                output.append("")
        
        # Recommended Next Steps
        if report.recommended_next_steps:
            output.append("━" * 100)
            output.append("RECOMMENDED NEXT STEPS")
            output.append("━" * 100)
            for i, step in enumerate(report.recommended_next_steps, 1):
                output.append(f"{i}. {step}")
            output.append("")
        
        output.append("=" * 100)
        
        return "\n".join(output)
    
    def format_summary(self, report: MarketIntelligenceReport) -> Dict[str, Any]:
        """
        Format a condensed summary of the report
        """
        return {
            'query': report.query,
            'domain': ', '.join(report.components.domain),
            'executive_summary': report.executive_summary,
            'top_companies': [p.get('name') for p in report.key_players[:5]],
            'top_trends': report.emerging_trends[:3],
            'total_companies_found': len(report.competitive_analysis.companies_mentioned),
            'total_articles': sum([
                len(report.companies_landscape.articles),
                len(report.technology_landscape.articles),
                len(report.business_models.articles),
                len(report.innovations.articles),
                len(report.market_trends.articles),
                len(report.competitive_analysis.articles)
            ])
        }

