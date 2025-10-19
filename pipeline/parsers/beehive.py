import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.common.exceptions import TimeoutException
import logging
from typing import List, Optional
from datetime import datetime
from core.models import RawArticle
from core.storage import StorageManager

class BeehiveScraper:
    def __init__(self, newsletter_id: str, url: str, config: dict):
        self.newsletter_id = newsletter_id
        self.url = url
        self.config = config
        self.headless = True
        self.driver = None
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_webdriver_instance(self):
        chrome_options = Options()

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--no-first-run")
        chrome_options.add_argument("--no-default-browser-check")
        chrome_options.add_argument("--disable-default-apps")
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-ipc-flooding-protection")

        try:
            # Use ChromeDriverManager to handle driver installation and path resolution
            print("Setting up ChromeDriver using ChromeDriverManager...")

            # Configure ChromeDriverManager to be more robust
            from selenium.webdriver.chrome.service import Service as ChromeService

            # Try to use a specific version and handle path issues
            try:
                # First try to get the latest chromedriver
                driver_path = ChromeDriverManager().install()
                print(f"ChromeDriver installed at: {driver_path}")

                # The driver_path might be a file or directory, we need to find the actual executable
                if driver_path:
                    if os.path.isfile(driver_path):
                        # It's already a file path to the executable
                        actual_driver_path = driver_path
                    else:
                        # It's a directory, find the chromedriver executable
                        print(f"Looking for chromedriver executable in directory: {driver_path}")

                        # Look for the actual chromedriver executable
                        actual_driver_path = None

                        # First, try to find the standard chromedriver executable names
                        standard_names = ['chromedriver', 'chromedriver.exe']
                        for name in standard_names:
                            potential_path = os.path.join(driver_path, name)
                            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                                actual_driver_path = potential_path
                                break

                        # If not found, search recursively for the actual executable
                        if not actual_driver_path:
                            for root, dirs, files in os.walk(driver_path):
                                for file in files:
                                    # Skip known non-executable files
                                    if any(file.upper().startswith(prefix) for prefix in ['THIRD_PARTY', 'LICENSE', 'README', 'NOTICE']):
                                        continue

                                    # Look for actual chromedriver executable (not notices or licenses)
                                    if (file.lower() in ['chromedriver', 'chromedriver.exe'] or
                                        (file.lower().startswith('chromedriver') and file.lower().endswith(('.exe', '-linux64', '-win32.exe', '-mac64')))):
                                        file_path = os.path.join(root, file)
                                        if os.path.isfile(file_path):
                                            actual_driver_path = file_path
                                            break
                                if actual_driver_path:
                                    break

                        # As a last resort, find any file that looks like chromedriver and is executable
                        if not actual_driver_path:
                            for root, dirs, files in os.walk(driver_path):
                                for file in files:
                                    # Skip known non-executable files
                                    if any(file.upper().startswith(prefix) for prefix in ['THIRD_PARTY', 'LICENSE', 'README', 'NOTICE']):
                                        continue

                                    file_path = os.path.join(root, file)
                                    if (os.path.isfile(file_path) and
                                        'chromedriver' in file.lower() and
                                        (os.access(file_path, os.X_OK) or file.endswith('.exe'))):
                                        actual_driver_path = file_path
                                        break
                                if actual_driver_path:
                                    break

                    if actual_driver_path and os.path.exists(actual_driver_path):
                        print(f"Found ChromeDriver executable at: {actual_driver_path}")
                        # Make sure it's executable
                        if not os.access(actual_driver_path, os.X_OK):
                            try:
                                os.chmod(actual_driver_path, 0o755)
                                print(f"Made ChromeDriver executable: {actual_driver_path}")
                            except Exception as e:
                                print(f"Warning: Could not make ChromeDriver executable: {e}")

                        service = ChromeService(actual_driver_path)
                        print(f"Using verified ChromeDriver at: {actual_driver_path}")
                    else:
                        print(f"Could not find ChromeDriver executable in {driver_path}")
                        raise Exception(f"ChromeDriver executable not found in {driver_path}")
                else:
                    raise Exception("ChromeDriverManager returned empty path")

            except Exception as e:
                print(f"Error with automatic driver management: {e}")
                print("Falling back to manual driver management...")

                # Fallback: try to use system chromedriver if available
                try:
                    print("Attempting to use system ChromeDriver...")
                    service = ChromeService()
                    print("Using system ChromeDriver...")
                except Exception as e2:
                    print(f"System ChromeDriver also failed: {e2}")

                    # Last resort: try to find chromedriver in common system locations
                    print("Trying to find ChromeDriver in common system locations...")
                    common_paths = [
                        "/usr/bin/chromedriver",
                        "/usr/local/bin/chromedriver",
                        "/opt/chromedriver",
                        "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe",
                        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
                    ]

                    found_path = None
                    for path in common_paths:
                        if os.path.exists(path) and os.path.isfile(path):
                            found_path = path
                            break

                    if found_path:
                        print(f"Found ChromeDriver at system location: {found_path}")
                        service = ChromeService(found_path)
                    else:
                        print("ChromeDriver not found in any common system location")
                        raise Exception(f"All ChromeDriver methods failed. Original error: {e}. System error: {e2}")

        except Exception as e:
            print(f"Error: ChromeDriver setup failed: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Make sure Google Chrome is installed on your system")
            print("2. Try running: pip install webdriver-manager --upgrade")
            print("3. Check if antivirus software is blocking chromedriver")
            print("4. Ensure you have sufficient disk space for driver download")
            print("5. Try clearing webdriver cache: pip cache remove webdriver-manager")
            print(f"\nError details: {str(e)}")
            raise e

        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            print("ChromeDriver initialized successfully")
            return driver
        except Exception as e:
            print(f"Failed to create Chrome WebDriver instance: {e}")
            print("\nAdditional troubleshooting:")
            print("- Chrome version might be incompatible with downloaded driver")
            print("- Try updating Chrome to the latest version")
            print("- Or try: pip install --upgrade selenium webdriver-manager")
            print(f"\nFull error: {str(e)}")
            raise e
    
    def scrape_single_article(self, url):
        if not self.driver:
            self.driver = self.create_webdriver_instance()
        
        try:
            self.driver.get(url)
            time.sleep(2) 
            
            # Look for content blocks 
            content_blocks = self.driver.find_element(By.ID, "content-blocks")
            all_text = content_blocks.text
            try:
                title = self.driver.find_element(By.TAG_NAME, "h1").text
            except Exception:
                title = ""

            try:
                date_elem = self.driver.find_element(
                By.CSS_SELECTOR,
                'span.text-wt-text-on-background[style="opacity:0.75;"]'
            )
                published_date = date_elem.text.strip()
            except Exception:
                published_date = None

            article = {
            "title": title or "Untitled",
            "url": url,
            "content": all_text,
            "published_date": published_date,
            "author": None,  # Optional, for compatibility with downstream models
        }
            self.logger.info(f"Successfully scraped article: {url}")
            return article
            
        except Exception as e:
            self.logger.error(f"Error scraping article {url}: {str(e)}")
            return None
    
    # ... existing code ...

    def scrape_newsletter_posts(self, base_url, max_pages=None, since=None):
        """
        Scrape newsletter posts, stopping when articles older than 'since' are found.

        """
        if not self.driver:
            self.driver = self.create_webdriver_instance()

        current_page = 1
        newsletter_data = []
        should_stop = False  # Flag for early stopping
        storage = StorageManager()

        while not should_stop:
            page_url = f"{base_url}/archive?page={current_page}"
            print(f"Scraping page: {page_url}")
            self.driver.get(page_url)
            time.sleep(2)

            links = self.driver.find_elements(By.CSS_SELECTOR, 'a[data-discover="true"][href^="/p/"]')
            hrefs = [link.get_attribute('href') for link in links]
            print(f"Found {len(hrefs)} articles on page {current_page}")

            # If no articles found, we've reached the end
            if len(hrefs) == 0:
                print("No articles found on page, stopping")
                break

            for href in hrefs:
                if storage.article_exists(href):
                    print(f"Found already scraped article {href}, stopping")
                    should_stop = True
                    break
                print(f"Scraping article {href}")
                article = self.scrape_single_article(href)
            
                if article:
                # Check date if filtering is enabled
                    if since and article.get('published_date'):
                        try:
                        # Parse the date string
                            article_date = datetime.strptime(
                            article['published_date'], 
                            "%b %d, %Y"
                        )
                        
                        # If article is older than 'since', stop scraping
                            if article_date <= since:
                                print(f"✓ Reached old article ({article_date}), stopping")
                                should_stop = True
                                break  # Stop processing this page
                        except Exception:
                            pass  # If date parsing fails, include the article
                
                    newsletter_data.append({
                        'url': href,
                    'text': article,
                    'scraped_at': time.time()
                })
            
                time.sleep(1)  # Rate limiting

        # Check stopping conditions
            if should_stop:
                print("Stopped due to date filter")
                break
        
            if max_pages and current_page >= max_pages:
                print("Reached max_pages limit")
                break
        
      
            current_page += 1

        return newsletter_data

    def fetch_articles(self, since: Optional[datetime] = None) -> List[RawArticle]:
        """
        Fetch articles from the newsletter and convert to RawArticle objects.
        This method is called by the pipeline.

        Args:
            since: Optional datetime - only fetch articles newer than this

        Returns:
            List of RawArticle objects
        """
        try:
            self.logger.info(f"Fetching articles from {self.url}")

            # Call the existing scrape method
            raw_data = self.scrape_newsletter_posts(
                base_url=self.url,
                max_pages=None,
                since=since
            )

            # Convert to RawArticle objects
            articles = []
            for item in raw_data:
                article_data = item.get('text', {})

                # Parse published_date if available
                published_date = None
                if article_data.get('published_date'):
                    try:
                        published_date = datetime.strptime(
                            article_data['published_date'],
                            "%b %d, %Y"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to parse date: {e}")

                # Create RawArticle object
                raw_article = RawArticle(
                    title=article_data.get('title', 'Untitled'),
                    url=article_data.get('url', ''),
                    content=article_data.get('content', ''),
                    published_date=published_date,
                    author=article_data.get('author')
                )

                articles.append(raw_article)

            self.logger.info(f"✓ Fetched {len(articles)} articles")
            return articles

        except Exception as e:
            self.logger.error(f"Error in fetch_articles: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _load_all_content(self):
        while True:
            try:
                load_more_button = WebDriverWait(self.driver, 3).until(
                    EC.element_to_be_clickable((
                        By.CSS_SELECTOR, 
                        'span[style="color:#222222"].text-lg.sm\\:text-xl.font-regular.wt-body-font'
                    ))
                )
                
                if load_more_button.text == "Load More":
                    load_more_button.click()
                    self.logger.info("Clicked 'Load More' button")
                    time.sleep(2)
                else:
                    self.logger.info("'Load More' button found but text doesn't match")
                    break
                    
            except TimeoutException:
                self.logger.info("'Load More' button not found. All content loaded")
                break
            except Exception as e:
                self.logger.error(f"Error loading more content: {str(e)}")
                break
    
    def close(self):
        """Close the WebDriver instance"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



import time
import json
from datetime import datetime

def test_single_article():
    return True

def test_all_articles():
    """Test scraping all newsletter posts"""
    print("\n Testing all articles scraping...")
    
    # Replace with your actual beehive blog URL
    beehive_base_url = "https://aibreakfast.beehiiv.com"  # Replace this!
    
    try:
        with BeehiveScraper(headless=False) as scraper:  # Set to False to see browser
            print(f"Testing all articles from: {beehive_base_url}")
            
            articles = scraper.scrape_newsletter_posts(beehive_base_url)
            
            if articles:
                print(f" Successfully scraped {len(articles)} articles!")
                
                # Show details about scraped articles
                for i, article in enumerate(articles[:3], 1):  # Show first 3
                    print(f"\nArticle {i}:")
                    print(f"  URL: {article['url']}")
                    print(f"  Text length: {len(article['text'])} characters")
                    print(f"  Scraped at: {datetime.fromtimestamp(article['scraped_at'])}")
                
                if len(articles) > 3:
                    print(f"  ... and {len(articles) - 3} more articles")
                
                # Save to file for inspection
                output_file = f"test_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, indent=2, ensure_ascii=False)
                print(f"\n Articles saved to: {output_file}")
                
                return True
            else:
                print(" No articles were scraped")
                return False
                
    except Exception as e:
        print(f" All articles test failed: {e}")
        return False


def main():
    """Run all tests"""
    print(" Starting comprehensive beehive scraper tests...\n")
    
    
    # Test 2: Single article test (needs real beehive URL)
    single_success = test_single_article()
    
    # Test 3: All articles test (needs real beehive URL)
    all_success = test_all_articles()
    
    # Summary
    print("\n" + "="*50)
    print(" TEST SUMMARY")
    print("="*50)
    print(f"Single article test: {' PASS' if single_success else ' FAIL'}")
    print(f"All articles test: {' PASS' if all_success else ' FAIL'}")
    
    if  single_success and all_success:
        print("\n All tests passed! Your scraper is working correctly.")
    else:
        print("\n  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()