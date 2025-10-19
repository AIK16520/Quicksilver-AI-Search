#!/usr/bin/env python3
"""
One-time script to fill Business and Tech columns in product_hunt_products table.
Analyzes existing products and updates them with business model and technology classifications.
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
from typing import Dict, Optional

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

class ProductHuntFieldFiller:
    """Fills Business and Tech columns for existing Product Hunt products"""

    def __init__(self):
        self.supabase = supabase_client

    def extract_business_model(self, product: Dict) -> Optional[str]:
        """Extract business model from product description"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._extract_business_model_simple(product)

        try:
            prompt = f"""Analyze this product and determine its primary business model.
Return ONLY the business model type, no explanations.

Product Name: {product['product_name']}
Overview: {product.get('overview', '')}
Description: {product.get('description', '')[:300]}

Common business models: SaaS, Enterprise Software, Data-as-a-Service, API Platform, Consulting, Open Source, Freemium, Marketplace, etc.

Business Model:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            business_model = response.choices[0].message.content.strip()
            return business_model if business_model else None

        except Exception as e:
            logger.error(f"Failed to extract business model: {e}")
            return self._extract_business_model_simple(product)

    def extract_technology(self, product: Dict) -> Optional[str]:
        """Extract primary technology from product description"""
        if not OPENAI_AVAILABLE or not openai_client:
            return self._extract_technology_simple(product)

        try:
            prompt = f"""Analyze this product and determine its primary technology focus.
Return ONLY the technology type, no explanations.

Product Name: {product['product_name']}
Overview: {product.get('overview', '')}
Description: {product.get('description', '')[:300]}

Common technologies: AI/ML, Data Analytics, Automation, Blockchain, Cloud, Mobile, Web, IoT, etc.

Technology:"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )

            technology = response.choices[0].message.content.strip()
            return technology if technology else None

        except Exception as e:
            logger.error(f"Failed to extract technology: {e}")
            return self._extract_technology_simple(product)

    def _extract_business_model_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based business model extraction"""
        text = f"{product['product_name']} {product.get('overview', '')} {product.get('description', '')}".lower()

        if any(word in text for word in ['saas', 'subscription', 'monthly', 'platform']):
            return 'SaaS Platform'
        elif any(word in text for word in ['enterprise', 'licensing', 'on-premise']):
            return 'Enterprise Software'
        elif any(word in text for word in ['api', 'data service', 'data provider']):
            return 'Data-as-a-Service'
        elif any(word in text for word in ['ai', 'machine learning', 'automation']):
            return 'AI-Powered Analytics'
        elif any(word in text for word in ['consulting', 'professional services']):
            return 'Consulting Services'
        elif any(word in text for word in ['open source', 'free']):
            return 'Open Source'
        else:
            return 'Other'

    def _extract_technology_simple(self, product: Dict) -> Optional[str]:
        """Simple rule-based technology extraction"""
        text = f"{product['product_name']} {product.get('overview', '')} {product.get('description', '')}".lower()

        if any(word in text for word in ['ai', 'artificial intelligence', 'machine learning', 'ml']):
            return 'AI/ML'
        elif any(word in text for word in ['data', 'analytics', 'dashboard', 'reporting']):
            return 'Data Analytics'
        elif any(word in text for word in ['automation', 'workflow', 'process']):
            return 'Automation'
        elif any(word in text for word in ['blockchain', 'crypto', 'web3']):
            return 'Blockchain'
        elif any(word in text for word in ['cloud', 'saas']):
            return 'Cloud/SaaS'
        elif any(word in text for word in ['mobile', 'ios', 'android']):
            return 'Mobile'
        else:
            return 'Other'

    def fill_product_fields(self):
        """Fill Business and Tech fields for all products in the database"""
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

                # Skip if already has non-null Business and Tech fields
                business_value = product.get('Business')
                tech_value = product.get('Tech')

                if (business_value and business_value.strip()) and (tech_value and tech_value.strip()):
                    skipped_count += 1
                    logger.debug(f"Skipping {product_name} - already has Business ({business_value}) and Tech ({tech_value}) fields")
                    continue

                logger.info(f"Processing: {product_name}")

                # Extract business model and technology
                business_model = self.extract_business_model(product)
                technology = self.extract_technology(product)

                # Update the product in database
                update_data = {}
                if business_model and business_model != business_value:
                    update_data['Business'] = business_model
                if technology and technology != tech_value:
                    update_data['Tech'] = technology

                if update_data:
                    try:
                        self.supabase.table('product_hunt_products').update(update_data).eq('id', product_id).execute()
                        updated_count += 1
                        logger.info(f"Updated {product_name}: Business={business_model}, Tech={technology}")
                    except Exception as e:
                        logger.error(f"Failed to update {product_name}: {e}")
                else:
                    skipped_count += 1
                    logger.debug(f"No changes needed for {product_name}")

            logger.info(f"Successfully updated {updated_count} products, skipped {skipped_count} products")

        except Exception as e:
            logger.error(f"Failed to fetch/update products: {e}")
            return False

        return True

def main():
    """Main function to run the field filler"""
    logger.info("Starting Product Hunt field filler...")

    filler = ProductHuntFieldFiller()

    success = filler.fill_product_fields()

    if success:
        logger.info("Field filling completed successfully!")

        # Delete this script file
        script_path = Path(__file__)
        try:
            script_path.unlink()
            logger.info(f"Deleted script file: {script_path}")
        except Exception as e:
            logger.warning(f"Could not delete script file: {e}")
    else:
        logger.error("Field filling failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
