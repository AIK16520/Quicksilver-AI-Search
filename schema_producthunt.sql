-- Drop existing table
DROP TABLE IF EXISTS product_hunt_products CASCADE;

-- Create new Product Hunt table with AI features
CREATE TABLE product_hunt_products (
    -- Primary identifier
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic product information
    product_name TEXT NOT NULL,
    producthunt_link TEXT UNIQUE,
    overview TEXT,
    description TEXT,
    product_link TEXT,
    
    -- AI-generated fields
    ai_description TEXT,
    embedding VECTOR(1536),  -- Full description embedding
    keywords TEXT[],  -- Array of keywords for search
    keyword_embedding VECTOR(1536),  -- Embedding of name + keywords for better search
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_producthunt_link ON product_hunt_products(producthunt_link);
CREATE INDEX idx_product_name ON product_hunt_products(product_name);
CREATE INDEX idx_scraped_at ON product_hunt_products(scraped_at DESC);
CREATE INDEX idx_embedding ON product_hunt_products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_keyword_embedding ON product_hunt_products USING ivfflat (keyword_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_keywords ON product_hunt_products USING GIN (keywords);

-- Add trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_product_hunt_updated_at 
    BEFORE UPDATE ON product_hunt_products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
