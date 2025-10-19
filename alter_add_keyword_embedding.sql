-- Add keyword_embedding column to existing product_hunt_products table
ALTER TABLE product_hunt_products 
ADD COLUMN keyword_embedding VECTOR(1536);

-- Create index for the new keyword_embedding column
CREATE INDEX idx_keyword_embedding ON product_hunt_products 
USING ivfflat (keyword_embedding vector_cosine_ops) WITH (lists = 100);

-- Verify the column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'product_hunt_products' 
AND column_name = 'keyword_embedding';


