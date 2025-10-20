import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  Container,
  Typography,
  Box,
  AppBar,
  Toolbar,
  Paper,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Divider,
  LinearProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SearchIcon from '@mui/icons-material/Search';
import axios from 'axios';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState({ stage: '', percent: 0 });
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [deepDiveData, setDeepDiveData] = useState(null);
  const [deepDiveLoading, setDeepDiveLoading] = useState({});
  const [expandedCompanies, setExpandedCompanies] = useState({});

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setDeepDiveData(null);

    try {
      // Market intelligence search with improved loading stages
      const stages = [
        { name: 'Analyzing query...', percent: 5 },
        { name: 'Searching companies...', percent: 15 },
        { name: 'Searching technology...', percent: 25 },
        { name: 'Searching business models...', percent: 35 },
        { name: 'Searching innovations...', percent: 45 },
        { name: 'Searching market trends...', percent: 55 },
        { name: 'Discovering competitors...', percent: 65 },
        { name: 'Extracting entities...', percent: 70 },
        { name: 'AI generating insights...', percent: 75 },
        { name: 'AI analyzing technologies...', percent: 80 },
        { name: 'AI extracting business models...', percent: 85 },
        { name: 'AI identifying market trends...', percent: 90 },
        { name: 'AI analyzing key players...', percent: 95 },
        { name: 'Finalizing report...', percent: 98 }
      ];

      // Simulate progressive loading (since backend doesn't stream yet)
      let currentStage = 0;
      const progressInterval = setInterval(() => {
        if (currentStage < stages.length - 1) {
          currentStage++;
          setLoadingProgress({
            stage: stages[currentStage].name,
            percent: stages[currentStage].percent
          });
        }
      }, 1800); // Update every 1.8 seconds for smoother progress

      setLoadingProgress({ stage: stages[0].name, percent: stages[0].percent });

      const searchResponse = await axios.post(`${API_BASE_URL}/market-intelligence`, {
        query: query,
        max_results_per_dimension: 8,
        include_ai_insights: true,
        format_type: 'api'
      });

      clearInterval(progressInterval);
      setLoadingProgress({ stage: 'Complete!', percent: 100 });

      setResults({ mode: 'new', data: searchResponse.data.data });
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during search');
    } finally {
      setLoading(false);
      // Reset progress after a short delay
      setTimeout(() => setLoadingProgress({ stage: '', percent: 0 }), 1000);
    }
  };

  const handleDeepDive = async (companyName, entity, entityType, context) => {
    // Toggle: if already expanded, collapse it
    if (expandedCompanies[companyName]) {
      setExpandedCompanies(prev => ({ ...prev, [companyName]: null }));
      return;
    }

    setDeepDiveLoading(prev => ({ ...prev, [companyName]: true }));
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/deep-dive`, {
        entity: entity,
        entity_type: entityType,
        context: context,
        max_articles: 15
      });
      
      setExpandedCompanies(prev => ({ ...prev, [companyName]: response.data.data }));
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during deep dive');
    } finally {
      setDeepDiveLoading(prev => ({ ...prev, [companyName]: false }));
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
               Quicksilver Search
            </Typography>
           
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h4" gutterBottom>
              Search System
            </Typography>
      

            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search for industry trends, company news, etc."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
              />
              <Button
                variant="contained"
                size="large"
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
              >
                {loading ? 'Searching...' : 'Search'}
              </Button>
            </Box>

          </Paper>

          {/* Loading Progress Bar */}
          {loading && loadingProgress.stage && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ width: '100%' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {loadingProgress.stage}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={loadingProgress.percent}
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  {loadingProgress.percent}% complete
                </Typography>
              </Box>
            </Paper>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {results && (
            <Box>
              {/* Market Landscape Section */}
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                  Market Landscape
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                {/* Current State of Market */}
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Current State of the Market
                </Typography>
                <Typography variant="body1" sx={{ mb: 3 }} component="div">
                  {(() => {
                    let marketState = results.data.executive_summary || "Analysis of current market conditions and trends.";

                    // Add business model insights from dimensions
                    if (results.data.dimensions) {
                      const businessModels = [];
                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.business_model_details && dimension.business_model_details.length > 0) {
                          businessModels.push(...dimension.business_model_details.slice(0, 2));
                        }
                      });

                      if (businessModels.length > 0) {
                        const cleanedModels = businessModels.slice(0, 2).map(model =>
                          model.replace(/\s*\(Innovation:.*?\)\s*/g, '').split(':')[0]
                        );
                        marketState += " Key business models in this space include: " + cleanedModels.join(", ") + ".";
                      }

                      // Add market insights
                      const insights = [];
                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.market_insights && dimension.market_insights.length > 0) {
                          insights.push(...dimension.market_insights.slice(0, 2));
                        }
                      });

                      if (insights.length > 0) {
                        marketState += " " + insights.slice(0, 2).join(" ");
                      }
                    }

                    // Bold company names in the text
                    // Match capitalized words that look like company names (2+ words starting with capitals, or single words with Corp/Inc/LLC/Technologies/etc)
                    const companyPattern = /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+(?:Inc|LLC|Corp|Corporation|Technologies|Systems|Solutions|Group|Holdings|Limited|Ltd))?)|\b([A-Z][a-z]+(?:\s+(?:Inc|LLC|Corp|Corporation|Technologies|Systems|Solutions|Group|Holdings|Limited|Ltd)))\b/g;

                    const parts = [];
                    let lastIndex = 0;
                    let match;

                    while ((match = companyPattern.exec(marketState)) !== null) {
                      // Add text before the match
                      if (match.index > lastIndex) {
                        parts.push(marketState.substring(lastIndex, match.index));
                      }
                      // Add bolded company name
                      parts.push(<strong key={match.index}>{match[0]}</strong>);
                      lastIndex = match.index + match[0].length;
                    }

                    // Add remaining text
                    if (lastIndex < marketState.length) {
                      parts.push(marketState.substring(lastIndex));
                    }

                    return parts.length > 0 ? parts : marketState;
                  })()}
                </Typography>

                {/* Discovered Technologies & Innovations - Only show technologies without specific companies */}
                <Typography variant="h6" gutterBottom>
                  Key Technologies
                </Typography>
                <Box sx={{ mb: 3 }}>
                  {(() => {
                    const technologyDetails = {};

                    // Extract detailed technology usage from all dimensions
                    if (results.data.dimensions) {
                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.technology_usage_details) {
                          Object.entries(dimension.technology_usage_details).forEach(([tech, usages]) => {
                            if (!technologyDetails[tech]) {
                              technologyDetails[tech] = {
                                usages: [],
                                companies: []
                              };
                            }
                            technologyDetails[tech].usages = technologyDetails[tech].usages.concat(usages);
                          });
                        }
                      });
                    }

                    // Filter to only show general technologies (no specific companies)
                    const generalTechnologies = Object.entries(technologyDetails).filter(([tech, data]) => {
                      // Only show if it's a general trend/technology, not company-specific
                      return data.usages.length > 0 && !data.usages[0].toLowerCase().includes('company');
                    }).slice(0, 6); // Limit to 6 general technologies

                    return generalTechnologies.length > 0 ? (
                      <Grid container spacing={2}>
                        {generalTechnologies.map(([tech, data], index) => (
                          <Grid item xs={12} sm={6} md={4} key={index}>
                            <Paper sx={{ p: 2, height: '100%' }}>
                              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                                {tech}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                {data.usages[0]}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        General technology trends will appear here. Specific company technologies are shown in the Companies & Products section below.
                      </Typography>
                    );
                  })()}
                </Box>

              </Paper>

              {/* Companies Section - Combined Product Hunt and Business Model Companies */}
              {(() => {
                // Extract Product Hunt products from all dimension results
                const phProducts = [];
                const regularCompanies = [];
                const businessModelCompanies = [];

                if (results.data.dimensions) {
                  Object.values(results.data.dimensions).forEach(dimension => {
                    if (dimension.articles) {
                      dimension.articles.forEach(article => {
                        if (article.source === 'product_hunt') {
                          phProducts.push({
                            name: article.title,
                            url: article.url,
                            producthunt_link: article.producthunt_link,
                            description: article.description,
                            business_model: article.Business,
                            moat: article.Moat,
                            type: 'product_hunt'
                          });
                        }
                      });
                    }

                    // Extract companies associated with business models
                    if (dimension.company_business_models) {
                      Object.entries(dimension.company_business_models).forEach(([companyName, models]) => {
                        // Check if this company isn't already in regularCompanies or phProducts
                        const alreadyExists = regularCompanies.some(c => c.name === companyName) ||
                                            phProducts.some(p => p.name === companyName);

                        if (!alreadyExists && models.length > 0) {
                          // Try to find the article URL for this company across ALL dimensions
                          let companyUrl = null;
                          
                          // First check current dimension
                          if (dimension.articles) {
                            const companyArticle = dimension.articles.find(article =>
                              article.title && article.title.toLowerCase().includes(companyName.toLowerCase())
                            );
                            if (companyArticle) {
                              companyUrl = companyArticle.url;
                            }
                          }
                          
                          // If not found, search across all other dimensions
                          if (!companyUrl && results.data.dimensions) {
                            Object.values(results.data.dimensions).forEach(otherDimension => {
                              if (otherDimension.articles && !companyUrl) {
                                const companyArticle = otherDimension.articles.find(article =>
                                  article.title && article.title.toLowerCase().includes(companyName.toLowerCase())
                                );
                                if (companyArticle) {
                                  companyUrl = companyArticle.url;
                                }
                              }
                            });
                          }

                          businessModelCompanies.push({
                            name: companyName,
                            context: `Operates using ${models.join(', ')} business model${models.length > 1 ? 's' : ''}`,
                            business_models: models,
                            url: companyUrl,
                            type: 'business_model_company'
                          });
                        }
                      });
                    }
                  });
                }

                // Add regular companies from interactive elements
                if (results.data.interactive_elements?.companies) {
                  results.data.interactive_elements.companies.forEach(company => {
                    regularCompanies.push({
                      name: company.name,
                      context: company.context,
                      mentioned_in: company.mentioned_in,
                      type: 'company'
                    });
                  });
                }

                // Remove duplicates from Product Hunt products
                const uniquePhProducts = phProducts.filter((product, index, self) =>
                  index === self.findIndex(p => p.name === product.name)
                );

                // Remove duplicates from business model companies
                const uniqueBusinessModelCompanies = businessModelCompanies.filter((company, index, self) =>
                  index === self.findIndex(c => c.name === company.name)
                );

                // Combine all companies and products
                const allCompanies = [...uniquePhProducts, ...uniqueBusinessModelCompanies];
                
                // Separate those with URLs and those without
                const withUrls = allCompanies.filter(company => company.url);
                const withoutUrls = allCompanies.filter(company => !company.url);
                
                // Shuffle the withUrls array for random ordering
                const shuffledWithUrls = withUrls.sort(() => Math.random() - 0.5);
                
                // Combine: shuffled with URLs first, then without URLs
                const finalCompanies = [...shuffledWithUrls, ...withoutUrls];

                return (finalCompanies.length > 0 || regularCompanies.length > 0) && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                      Companies and Products
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                    {/* Combined Companies and Products */}
                    {finalCompanies.length > 0 && (
                      <Box sx={{ mb: 3 }}>
                        <Grid container spacing={2}>
                          {finalCompanies.map((company, index) => (
                            <Grid item xs={12} md={6} key={`company-${index}`}>
                              <Card
                                sx={{
                                  height: '100%',
                                  cursor: company.url ? 'pointer' : 'default',
                                  '&:hover': company.url ? {
                                    boxShadow: 3,
                                    transform: 'translateY(-2px)',
                                    transition: 'all 0.2s ease-in-out'
                                  } : {}
                                }}
                                onClick={() => company.url && window.open(company.url, '_blank')}
                              >
                                <CardContent>
                                  <Typography variant="h6" sx={{ mb: 2 }}>{company.name}</Typography>

                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    {company.description || company.context || "Emerging product in the market"}
                                  </Typography>

                                  {/* Business Model for Product Hunt products */}
                                  {company.business_model && (
                                    <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                        💼 Business Model: {company.business_model}
                                      </Typography>
                                    </Box>
                                  )}

                                  {/* Competitive Moat for Product Hunt products */}
                                  {company.moat && (
                                    <Box sx={{ mb: 2, p: 2, bgcolor: 'blue.50', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                        🏰 Competitive Moat
                                      </Typography>
                                      <Typography variant="body2" color="text.secondary">
                                        {company.moat}
                                      </Typography>
                                    </Box>
                                  )}

                                </CardContent>
                              </Card>
                            </Grid>
                          ))}
                        </Grid>
                      </Box>
                    )}

                    {/* Regular Companies */}
                    {regularCompanies.length > 0 && (
                      <Box>
                        <Typography variant="h6" gutterBottom>
                          Established Companies
                        </Typography>
                  <Grid container spacing={2}>
                          {regularCompanies.map((company, index) => {
                            const deepDiveAction = company.actions?.find(a => a.type === 'deep_dive');
                      const isExpanded = !!expandedCompanies[company.name];
                      const isLoading = !!deepDiveLoading[company.name];
                      
                      return (
                              <Grid item xs={12} md={6} key={`company-${index}`}>
                                <Card sx={{ height: '100%' }}>
                            <CardContent>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                                      <Typography variant="h6" sx={{ flex: 1 }}>{company.name}</Typography>
                                      {deepDiveAction && (
                                <Button
                                  variant={isExpanded ? 'outlined' : 'contained'}
                                  size="small"
                                  disabled={isLoading}
                                  onClick={() => {
                                    if (deepDiveAction) {
                                      handleDeepDive(
                                        company.name,
                                        deepDiveAction.params.entity,
                                        deepDiveAction.params.entity_type,
                                        deepDiveAction.params.context
                                      );
                                    }
                                  }}
                                  startIcon={isLoading ? <CircularProgress size={16} /> : null}
                                >
                                  {isLoading ? 'Loading...' : isExpanded ? 'Close' : 'Deep Dive'}
                                </Button>
                                      )}
                              </Box>

                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                      {company.context}
                                    </Typography>

                                    <Chip
                                      label={`Mentioned in: ${company.mentioned_in?.join(', ') || 'analysis'}`}
                                      size="small"
                                      variant="outlined"
                                    />

                              {/* Inline Deep Dive Results */}
                              {isExpanded && expandedCompanies[company.name] && (
                                <Box sx={{ mt: 3, pt: 3, borderTop: '1px solid #e0e0e0' }}>
                                  <Typography variant="h6" gutterBottom color="primary">
                                    Deep Dive Analysis
                                  </Typography>
                                  
                                  <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
                                    {expandedCompanies[company.name].overview}
                                  </Typography>

                                  <Grid container spacing={2}>
                                          {expandedCompanies[company.name].technologies_used && expandedCompanies[company.name].technologies_used.length > 0 && (
                                            <Grid item xs={12}>
                                              <Typography variant="subtitle2" gutterBottom>Technologies:</Typography>
                                              <Typography variant="body2">{expandedCompanies[company.name].technologies_used.join(', ')}</Typography>
                                      </Grid>
                                    )}

                                    {expandedCompanies[company.name].business_model && (
                                            <Grid item xs={12}>
                                        <Typography variant="subtitle2" gutterBottom>Business Model:</Typography>
                                        <Typography variant="body2">{expandedCompanies[company.name].business_model}</Typography>
                                      </Grid>
                                    )}

                                          {expandedCompanies[company.name].competitors && expandedCompanies[company.name].competitors.length > 0 && (
                                      <Grid item xs={12}>
                                              <Typography variant="subtitle2" gutterBottom>Competitors:</Typography>
                                              <Typography variant="body2">{expandedCompanies[company.name].competitors.join(', ')}</Typography>
                                      </Grid>
                                    )}
                                  </Grid>
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                              </Box>
                    )}
                  </Paper>
                );
              })()}
            </Box>
          )}

          {/* Deep dive now shown inline in company cards */}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;

