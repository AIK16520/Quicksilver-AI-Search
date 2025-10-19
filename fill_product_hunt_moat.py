#!/usr/bin/env python3
"""
One-time script to fill Moat column in product_hunt_products table.
Analyzes existing products and updates them with competitive moat analysis.
Deletes itself after successful completion.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

# Import required modules
from core.config import supabase_client
import re
from typing import Dict, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

class ProductHuntMoatFiller:
    """Fills Moat column for existing Product Hunt products"""

    def __init__(self):
        self.supabase = supabase_client

    def generate_moat(self, product: Dict) -> Optional[str]:
        """Generate competitive moat analysis for the product"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._generate_moat_simple(product)

        try:
            prompt = f"""Analyze this financial analytics product and identify its competitive moat/advantage.

Product Name: {product['product_name']}
Overview: {product.get('overview', '')}
Description: {product.get('description', '')[:300]}
Business Model: {product.get('Business', 'Unknown')}
Technology: {product.get('Tech', 'Unknown')}

Identify the key competitive advantages that make this product defensible:

1. Technology/IP advantages (unique algorithms, data sources, etc.)
2. Network effects (user base, data advantages, etc.)
3. Brand/reputation advantages
4. Cost advantages
5. Regulatory/compliance advantages
6. Switching costs for customers

Return a concise 2-3 sentence description of their competitive moat, focusing on how they make money and maintain their advantage.

Moat Description:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            moat = response.choices[0].message.content.strip()
            return moat if moat else None

        except Exception as e:
            logger.error(f"Failed to generate moat: {e}")
            return self._generate_moat_simple(product)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if not OPENAI_AVAILABLE or not openai_client:
            return None

        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _generate_moat_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based moat generation"""
        text = f"{product['product_name']} {product.get('overview', '')} {product.get('description', '')}".lower()
        business_model = product.get('Business', '').lower()
        technology = product.get('Tech', '').lower()

        # Generate moat based on business model and technology
        if 'saas' in business_model and 'ai' in technology:
            return 'Proprietary AI algorithms and machine learning models provide predictive accuracy advantages, while subscription-based SaaS model creates recurring revenue and customer lock-in through data integration.'
        elif 'data' in business_model.lower():
            return 'Exclusive data partnerships and proprietary collection methods create data quality advantages, with API-based delivery enabling seamless integration and customer retention.'
        elif 'enterprise' in business_model.lower():
            return 'Deep regulatory compliance expertise and complex integration capabilities create high switching costs, while enterprise licensing provides stable, high-value revenue streams.'
        elif 'ai' in technology.lower():
            return 'Advanced machine learning models and continuous algorithm improvement provide accuracy advantages, with platform effects creating data network advantages over time.'
        else:
            return 'Innovative technology stack and market positioning provide differentiation, with scalable infrastructure enabling cost advantages at scale.'

    def fill_product_moat(self):
        """Fill Moat field for all products in the database"""
        logger.info("Fetching all products from database...")

        try:
            # Fetch all products
            response = self.supabase.table('product_hunt_products').select('*').execute()
            products = response.data

            logger.info(f"Found {len(products)} products to process")

            updated_count = 0
            skipped_count = 0

            for product in products:
                product_id = product['id']
                product_name = product['product_name']

                # Skip if already has non-null Moat field
                moat_value = product.get('moat')
                if moat_value and moat_value.strip():
                    skipped_count += 1
                    logger.debug(f"Skipping {product_name} - already has Moat field")
                    continue

                logger.info(f"Processing: {product_name}")

                # Generate moat analysis and embedding
                moat = self.generate_moat(product)

                # Generate moat embedding if moat is available
                moat_embedding = None
                if moat and OPENAI_AVAILABLE and openai_client:
                    try:
                        moat_embedding = self.generate_embedding(moat)
                        logger.info(f"Generated moat embedding for {product_name}")
                    except Exception as e:
                        logger.warning(f"Failed to generate moat embedding for {product_name}: {e}")

                # Update the product in database
                update_data = {}
                if moat:
                    update_data['moat'] = moat
                if moat_embedding:
                    update_data['moat_embedding'] = moat_embedding

                if update_data:
                    try:
                        self.supabase.table('product_hunt_products').update(update_data).eq('id', product_id).execute()
                        updated_count += 1
                        logger.info(f"Updated {product_name} with moat: {moat[:50]}...")
                    except Exception as e:
                        logger.error(f"Failed to update {product_name}: {e}")
                else:
                    skipped_count += 1
                    logger.debug(f"No moat data generated for {product_name}")

            logger.info(f"Successfully updated {updated_count} products, skipped {skipped_count} products")

        except Exception as e:
            logger.error(f"Failed to fetch/update products: {e}")
            return False

        return True

def main():
    """Main function to run the moat filler"""
    logger.info("Starting Product Hunt moat filler...")

    filler = ProductHuntMoatFiller()

    success = filler.fill_product_moat()

    if success:
        logger.info("Moat filling completed successfully!")

        # Delete this script file
        script_path = Path(__file__)
        try:
            script_path.unlink()
            logger.info(f"Deleted script file: {script_path}")
        except Exception as e:
            logger.warning(f"Could not delete script file: {e}")
    else:
        logger.error("Moat filling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
