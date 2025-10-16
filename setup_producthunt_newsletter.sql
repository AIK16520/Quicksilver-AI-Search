-- Add Product Hunt to the newsletters table
-- This allows Product Hunt to run automatically with the pipeline

INSERT INTO newsletters (
    name,
    url,
    source_type
) VALUES (
    'Product Hunt Top Products',
    'https://www.producthunt.com',
    'producthunt'
)
ON CONFLICT (url) DO UPDATE SET
    source_type = EXCLUDED.source_type;

-- Verify it was added
SELECT id, name, source_type, url 
FROM newsletters 
WHERE source_type = 'producthunt';

