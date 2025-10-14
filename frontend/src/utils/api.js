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

  // Health check
  healthCheck: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  }
};

export default searchAPI;
