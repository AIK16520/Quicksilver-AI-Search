#!/usr/bin/env python3
"""
Test script to verify processor and chunking works before running full pipeline.
"""

import logging
from processor import Processor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_processor():
    """Test the processor with a sample article"""

    print("\n" + "="*70)
    print("Testing Processor (OpenAI API)")
    print("="*70 + "\n")

    # Sample article data
    test_article = {
        'title': 'OpenAI Launches New GPT-4 Model',
        'url': 'https://example.com/test',
        'content': '''
        OpenAI has announced the launch of GPT-4, their most advanced language model yet.
        The new model shows significant improvements in reasoning and comprehension.

        Microsoft has invested heavily in OpenAI and will integrate the technology into their products.
        Google is also working on competing AI models through their DeepMind division.

        The release marks a major milestone in artificial intelligence development.
        Industry experts predict widespread adoption across multiple sectors.
        ''',
        'published_date': '2024-03-15',
        'author': 'Test Author'
    }

    try:
        # Initialize processor
        print("1. Initializing processor...")
        processor = Processor()
        print("   ✓ Processor initialized\n")

        # Process article
        print("2. Processing test article...")
        print(f"   Title: {test_article['title']}")
        print(f"   Content length: {len(test_article['content'])} characters\n")

        processed_article, chunks = processor.process_article(test_article)

        if not processed_article:
            print("   ✗ Processing failed - no processed article returned")
            return False

        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        print(f"\n✓ Summary generated:")
        print(f"  {processed_article.summary}\n")

        print(f"✓ Companies extracted: {processed_article.company_names}\n")

        print(f"✓ Summary embedding: {len(processed_article.summary_embedding)} dimensions\n")

        print(f"✓ Chunks created: {len(chunks)} total")

        # Count by level
        para_chunks = [c for c in chunks if c.chunk_level == 'paragraph']
        sent_chunks = [c for c in chunks if c.chunk_level == 'sentence']

        print(f"  - Paragraph chunks: {len(para_chunks)}")
        print(f"  - Sentence chunks: {len(sent_chunks)}")

        # Show first chunk as example
        if chunks:
            first_chunk = chunks[0]
            print(f"\n✓ Example chunk (first {first_chunk.chunk_level}):")
            print(f"  Text: {first_chunk.chunk_text[:100]}...")
            print(f"  Embedding: {len(first_chunk.embedding)} dimensions")

        print("\n" + "="*70)
        print("TEST PASSED ✓")
        print("="*70)
        print("\nYour processor is working correctly!")
        print("You can now run: python run.py")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "="*70)
        print("TROUBLESHOOTING")
        print("="*70)
        print("1. Make sure OPENAI_API_KEY is set in your .env file")
        print("2. Run: pip install --upgrade openai")
        print("3. Check that you have API credits available")
        print("="*70 + "\n")

        return False

if __name__ == "__main__":
    test_processor()
