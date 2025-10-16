# entity_extraction.py

"""
Extract and enrich entities (companies, partnerships, technologies) from search results
Makes them interactive and searchable
"""

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from openai import OpenAI
from core.config import OPENAI_API_KEY, GPT_MODEL

logger = logging.getLogger("entity_extraction")

@dataclass
class ExtractedEntity:
    """Represents an entity found in search results"""
    name: str
    type: str  # "company", "partnership", "technology", "trend"
    context: str  # Where/how it was mentioned
    mentioned_in: List[str]  # Which dimensions mentioned it
    confidence: float
    
    # Interactive elements
    actions: List[Dict[str, str]]
    vague: bool = False  # True for "several AI startups"
    resolution_query: Optional[str] = None  # Query to resolve vague entities


class EntityExtractionService:
    """
    Extracts entities from market intelligence reports and makes them interactive
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        
        # Common entity patterns
        self.company_indicators = [
            'announced', 'launched', 'released', 'acquired', 'partnered',
            'company', 'startup', 'Inc.', 'Corp.', 'Ltd.'
        ]
        
        self.vague_patterns = [
            r'several .+ (startups|companies)',
            r'multiple .+ (firms|companies)',
            r'various .+ (players|vendors)',
            r'many .+ (companies|startups)',
            r'a number of .+ (companies|organizations)'
        ]
    
    def extract_entities(
        self,
        report_data: Dict,
        original_query: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Extract all entities from a market intelligence report
        
        Args:
            report_data: The formatted market intelligence report
            original_query: The original search query for context
            
        Returns:
            Dictionary of entities by type
        """
        logger.info("Extracting entities from market intelligence report")
        
        entities = {
            'companies': [],
            'partnerships': [],
            'technologies': [],
            'trends': [],
            'vague_mentions': []
        }
        
        # Extract from executive summary
        if report_data.get('executive_summary'):
            summary_entities = self._extract_from_text(
                report_data['executive_summary'],
                'executive_summary',
                original_query
            )
            self._merge_entities(entities, summary_entities)
        
        # Extract from key players
        if report_data.get('key_players'):
            for player in report_data['key_players']:
                company = ExtractedEntity(
                    name=player.get('name', ''),
                    type='company',
                    context=player.get('description', ''),
                    mentioned_in=['key_players'],
                    confidence=0.9,
                    actions=self._generate_company_actions(player.get('name', ''), original_query),
                    vague=False
                )
                entities['companies'].append(company)
        
        # Extract from emerging trends
        if report_data.get('emerging_trends'):
            for trend in report_data['emerging_trends']:
                trend_entities = self._extract_from_text(trend, 'trends', original_query)
                self._merge_entities(entities, trend_entities)
        
        # Extract from each dimension
        if report_data.get('dimensions'):
            for dim_name, dim_data in report_data['dimensions'].items():
                # From key findings
                for finding in dim_data.get('key_findings', []):
                    dim_entities = self._extract_from_text(finding, dim_name, original_query)
                    self._merge_entities(entities, dim_entities)
                
                # From articles
                for article in dim_data.get('articles', [])[:5]:  # Top 5 articles per dimension
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    article_entities = self._extract_from_text(text, dim_name, original_query)
                    self._merge_entities(entities, article_entities)
        
        # Deduplicate and rank
        entities = self._deduplicate_entities(entities)
        
        logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
        return entities
    
    def _extract_from_text(
        self,
        text: str,
        source: str,
        original_query: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """Extract entities from a piece of text"""
        
        if self.openai_client:
            return self._ai_extraction(text, source, original_query)
        else:
            return self._pattern_extraction(text, source, original_query)
    
    def _ai_extraction(
        self,
        text: str,
        source: str,
        original_query: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """Use AI to extract entities"""
        
        prompt = f"""Extract key entities from this text. Focus on companies, partnerships, and specific technologies mentioned.

Text: {text}

Return ONLY valid JSON:
{{
  "companies": [
    {{"name": "Company Name", "context": "what they're doing"}},
    ...
  ],
  "partnerships": [
    {{"description": "X partnered with Y", "partners": ["X", "Y"]}},
    ...
  ],
  "technologies": [
    {{"name": "Technology Name", "context": "how it's used"}},
    ...
  ],
  "vague_mentions": [
    {{"text": "several AI startups", "resolution_query": "specific search to find them"}},
    ...
  ]
}}

For vague mentions like "several AI startups" or "multiple companies", create a resolution_query that would find the specific entities."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an entity extraction expert. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            import json
            result = json.loads(response.choices[0].message.content.strip())
            
            entities = {
                'companies': [],
                'partnerships': [],
                'technologies': [],
                'vague_mentions': []
            }
            
            # Process companies
            for company in result.get('companies', []):
                entities['companies'].append(ExtractedEntity(
                    name=company['name'],
                    type='company',
                    context=company.get('context', ''),
                    mentioned_in=[source],
                    confidence=0.8,
                    actions=self._generate_company_actions(company['name'], original_query),
                    vague=False
                ))
            
            # Process partnerships
            for partnership in result.get('partnerships', []):
                entities['partnerships'].append(ExtractedEntity(
                    name=partnership['description'],
                    type='partnership',
                    context=partnership['description'],
                    mentioned_in=[source],
                    confidence=0.7,
                    actions=self._generate_partnership_actions(partnership, original_query),
                    vague=False
                ))
            
            # Process technologies
            for tech in result.get('technologies', []):
                entities['technologies'].append(ExtractedEntity(
                    name=tech['name'],
                    type='technology',
                    context=tech.get('context', ''),
                    mentioned_in=[source],
                    confidence=0.7,
                    actions=self._generate_tech_actions(tech['name'], original_query),
                    vague=False
                ))
            
            # Process vague mentions
            for vague in result.get('vague_mentions', []):
                entities['vague_mentions'].append(ExtractedEntity(
                    name=vague['text'],
                    type='vague_mention',
                    context=text,
                    mentioned_in=[source],
                    confidence=0.6,
                    actions=[{
                        'type': 'resolve',
                        'label': f"Find {vague['text']}",
                        'query': vague['resolution_query']
                    }],
                    vague=True,
                    resolution_query=vague['resolution_query']
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
            return self._pattern_extraction(text, source, original_query)
    
    def _pattern_extraction(
        self,
        text: str,
        source: str,
        original_query: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """Fallback pattern-based extraction"""
        
        entities = {
            'companies': [],
            'partnerships': [],
            'technologies': [],
            'vague_mentions': []
        }
        
        # Simple pattern: Find capitalized multi-word phrases
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(pattern, text)
        
        for match in matches[:5]:  # Limit to avoid noise
            if any(indicator in text.lower() for indicator in self.company_indicators):
                entities['companies'].append(ExtractedEntity(
                    name=match,
                    type='company',
                    context=text[:100],
                    mentioned_in=[source],
                    confidence=0.5,
                    actions=self._generate_company_actions(match, original_query),
                    vague=False
                ))
        
        # Check for vague patterns
        for pattern in self.vague_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities['vague_mentions'].append(ExtractedEntity(
                    name=match,
                    type='vague_mention',
                    context=text,
                    mentioned_in=[source],
                    confidence=0.6,
                    actions=[{
                        'type': 'resolve',
                        'label': f"Find specific companies",
                        'query': f"specific companies {match}"
                    }],
                    vague=True,
                    resolution_query=f"specific {match}"
                ))
        
        return entities
    
    def _generate_company_actions(self, company_name: str, context: str) -> List[Dict[str, str]]:
        """Generate interactive actions for a company"""
        return [
            {
                'type': 'deep_dive',
                'label': f'Deep Dive on {company_name}',
                'endpoint': '/deep-dive',
                'params': {
                    'entity': company_name,
                    'entity_type': 'company',
                    'context': context
                }
            },
            {
                'type': 'recent_news',
                'label': 'Recent News',
                'endpoint': '/deep-dive',
                'params': {
                    'entity': f'{company_name} recent news',
                    'entity_type': 'company',
                    'context': f'{context} - recent developments'
                }
            },
            {
                'type': 'competitors',
                'label': 'Find Competitors',
                'endpoint': '/analyze-competitors',
                'params': {
                    'query': company_name
                }
            }
        ]
    
    def _generate_partnership_actions(self, partnership: Dict, context: str) -> List[Dict[str, str]]:
        """Generate actions for a partnership"""
        description = partnership.get('description', '')
        partners = partnership.get('partners', [])
        
        actions = []
        
        # Add action to find details about the partnership
        actions.append({
            'type': 'resolve_partnership',
            'label': 'Find Partnership Details',
            'endpoint': '/deep-dive',
            'params': {
                'entity': description,
                'entity_type': 'partnership',
                'context': context
            }
        })
        
        # Add actions for each partner
        for partner in partners[:2]:  # Limit to 2 partners
            actions.append({
                'type': 'deep_dive',
                'label': f'Deep Dive on {partner}',
                'endpoint': '/deep-dive',
                'params': {
                    'entity': partner,
                    'entity_type': 'company',
                    'context': context
                }
            })
        
        return actions
    
    def _generate_tech_actions(self, tech_name: str, context: str) -> List[Dict[str, str]]:
        """Generate actions for a technology"""
        return [
            {
                'type': 'tech_deep_dive',
                'label': f'Explore {tech_name}',
                'endpoint': '/deep-dive',
                'params': {
                    'entity': tech_name,
                    'entity_type': 'technology',
                    'context': context
                }
            },
            {
                'type': 'companies_using',
                'label': 'Companies Using This',
                'endpoint': '/search',
                'params': {
                    'query': f'companies using {tech_name}'
                }
            }
        ]
    
    def _merge_entities(self, target: Dict, source: Dict):
        """Merge entities from source into target"""
        for entity_type, entities in source.items():
            if entity_type in target:
                target[entity_type].extend(entities)
    
    def _deduplicate_entities(self, entities: Dict) -> Dict:
        """Remove duplicate entities and merge their contexts"""
        for entity_type, entity_list in entities.items():
            seen = {}
            
            for entity in entity_list:
                key = entity.name.lower()
                
                if key in seen:
                    # Merge: add mentioned_in sources
                    seen[key].mentioned_in.extend(entity.mentioned_in)
                    seen[key].mentioned_in = list(set(seen[key].mentioned_in))
                    # Use higher confidence
                    seen[key].confidence = max(seen[key].confidence, entity.confidence)
                else:
                    seen[key] = entity
            
            entities[entity_type] = list(seen.values())
        
        # Sort by confidence
        for entity_type in entities:
            entities[entity_type].sort(key=lambda x: x.confidence, reverse=True)
        
        return entities

