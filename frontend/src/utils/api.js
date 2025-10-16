import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const searchAPI = {
  // Detect query type and intent
  detectQueryType: async (query) => {
    const response = await axios.post(`${API_BASE_URL}/query-type`, { query });
    return response.data;
  },

  // Build search context
  buildContext: async (mode, entity) => {
    const response = await axios.post(`${API_BASE_URL}/context`, { mode, entity });
    return response.data;
  },

  // Perform search
  search: async (query, options = {}) => {
    const { includeWeb = true, limit = 10, generateInsights = true } = options;
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query,
      include_web: includeWeb,
      limit,
      generate_insights: generateInsights
    });
    return response.data;
  },

  // Get portfolio companies
  getPortfolioCompanies: async () => {
    const response = await axios.get(`${API_BASE_URL}/portfolio-companies`);
    return response.data;
  },

  // Get portfolio maps info
  getPortfolioMaps: async () => {
    const response = await axios.get(`${API_BASE_URL}/portfolio-maps`);
    return response.data;
  },

  // Enhanced competitor analysis with discovery
  analyzeCompetitors: async (query, options = {}) => {
    const {
      includeDiscovery = true,
      config = null,
      searchDepth = "moderate"
    } = options;

    const response = await axios.post(`${API_BASE_URL}/analyze-competitors`, {
      query,
      include_discovery: includeDiscovery,
      config,
      search_depth: searchDepth
    });
    return response.data;
  },

  // Multi-dimensional market intelligence analysis
  analyzeMarketIntelligence: async (query, options = {}) => {
    const {
      maxResultsPerDimension = 8,
      includeAiInsights = true,
      formatType = "api"
    } = options;

    const response = await axios.post(`${API_BASE_URL}/market-intelligence`, {
      query,
      max_results_per_dimension: maxResultsPerDimension,
      include_ai_insights: includeAiInsights,
      format_type: formatType
    });
    return response.data;
  },

  // Deep dive on specific entity (company, partnership, technology)
  deepDive: async (entity, options = {}) => {
    const {
      entityType = "company",
      context = "",
      maxArticles = 15
    } = options;

    const response = await axios.post(`${API_BASE_URL}/deep-dive`, {
      entity,
      entity_type: entityType,
      context,
      max_articles: maxArticles
    });
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  }
};

export default searchAPI;

