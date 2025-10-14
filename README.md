# Quicksilver - VC Search Intelligence Platform

A comprehensive search system designed for venture capitalists, providing intelligent search with portfolio company context and competitive intelligence.

## Features

- **Intelligent Query Detection**: Automatically detects whether you're asking about specific companies, industries, or general portfolio monitoring
- **Portfolio-Aware Search**: Uses your portfolio company data to provide contextually relevant results
- **Live Web Integration**: Combines curated database results with live web search
- **AI-Powered Insights**: Generates VC-focused insights and strategic recommendations
- **REST API**: Clean API interface for integration with frontend applications

## Project Structure

```
Quicksilver/
├── core/                 # Core application logic
│   ├── app.py           # FastAPI application
│   ├── config.py        # Configuration and environment setup
│   ├── models.py        # Data models
│   ├── storage.py       # Database operations
│   ├── matching.py      # Article matching logic
│   └── requirements.txt # Python dependencies
├── search/              # Search service
│   ├── search.py        # Main search service
│   └── test_context_search.py # Search testing utilities
├── data/                # Data files and mappings
│   ├── build_portfolio_maps.py # Portfolio mapping builder
│   ├── all_industries.json     # Industry mappings
│   ├── all_keywords.json       # Keyword mappings
│   └── industry_to_companies.json # Industry-company mappings
├── pipeline/            # Data processing pipeline
│   ├── pipeline.py      # Main pipeline
│   ├── processor.py     # Article processing
│   └── run.py          # Pipeline runner
├── frontend/           # React frontend
│   ├── public/         # Static assets
│   ├── src/           # React source code
│   └── package.json   # Frontend dependencies
└── docs/              # Documentation
```

## Quick Start

### 1. Backend API Setup

```bash
# Navigate to the core directory
cd core

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables in .env file:
# SUPABASE_URL=your_supabase_url
# SUPABASE_KEY=your_supabase_key
# OPENAI_API_KEY=your_openai_key (optional)
# BRAVE_API_KEY=your_brave_api_key (optional)

# Run the API server
python app.py
```

The API will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Search Operations
- `POST /search` - Perform intelligent search
- `POST /query-type` - Detect query intent
- `POST /context` - Build search context

### Information
- `GET /portfolio-companies` - Get portfolio companies
- `GET /portfolio-maps` - Get mapping information
- `GET /health` - Health check
- `GET /` - API information

## Usage Examples

### Python API Usage

```python
from core.app import search_service

# Detect query type
mode, entity = search_service._detect_query_type("food tech trends")

# Perform search
results = search_service.search(
    query="AI infrastructure trends",
    include_web=True,
    limit=10
)
```

### Frontend Usage

```javascript
import searchAPI from './utils/api';

// Detect query type
const queryType = await searchAPI.detectQueryType("food tech updates");

// Perform search
const results = await searchAPI.search("AI industry trends", {
  includeWeb: true,
  limit: 10
});
```

## Configuration

The system uses environment variables for configuration. Create a `.env` file in the core directory:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
BRAVE_API_KEY=your_brave_search_api_key
```

## Development

### Running Tests

```bash
# Test search functionality
cd search
python test_context_search.py
```

### Building Portfolio Maps

```bash
# Build keyword and industry mappings from portfolio data
cd data
python build_portfolio_maps.py
```

## Deployment

### API Deployment

The FastAPI application can be deployed using:
- **Docker**: `docker build -t quicksilver-api . && docker run -p 8000:8000 quicksilver-api`
- **Heroku**: Add a `Procfile` with `web: uvicorn core.app:app --host 0.0.0.0 --port $PORT`
- **AWS/GCP**: Use serverless deployment options

### Frontend Deployment

The React frontend can be deployed to:
- **Netlify**: `npm run build && netlify deploy --prod --dir=build`
- **Vercel**: `npm run build && vercel --prod`
- **AWS S3**: `npm run build && aws s3 sync build/ s3://your-bucket`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
