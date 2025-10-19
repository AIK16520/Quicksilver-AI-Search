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
  const [includeWeb, setIncludeWeb] = useState(true);
  const [searchMode, setSearchMode] = useState('new'); // 'old' or 'new'
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState({ stage: '', current: 0, total: 0, percent: 0 });
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
      if (searchMode === 'old') {
        // OLD SEARCH - Portfolio-focused with insights
        setLoadingProgress({ stage: 'Searching database and web...', current: 1, total: 3, percent: 33 });
        
      const searchResponse = await axios.post(`${API_BASE_URL}/search`, {
        query: query,
        include_web: includeWeb,
        limit: 7,
        generate_insights: true
      });

        setLoadingProgress({ stage: 'Generating insights...', current: 2, total: 3, percent: 66 });
        
        // Simulate slight delay to show progress
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setLoadingProgress({ stage: 'Complete!', current: 3, total: 3, percent: 100 });
        setResults({ mode: 'old', data: searchResponse.data });
      } else {
        // NEW SEARCH - Market intelligence with interactive elements
        const stages = [
          'Analyzing query...',
          'Searching companies...',
          'Searching technology...',
          'Searching business models...',
          'Searching innovations...',
          'Searching market trends...',
          'Discovering competitors...',
          'Extracting entities...',
          'AI generating insights (this takes a moment)...',
          'AI analyzing key players...',
          'AI identifying trends...',
          'Finalizing report...'
        ];

        // Simulate progressive loading (since backend doesn't stream yet)
        let currentStage = 0;
        const progressInterval = setInterval(() => {
          if (currentStage < stages.length - 2) {  // Stop before "Finalizing"
            currentStage++;
            setLoadingProgress({
              stage: stages[currentStage],
              current: currentStage + 1,
              total: stages.length,
              percent: Math.round(((currentStage + 1) / stages.length) * 100)
            });
          }
        }, 2500); // Update every 2.5 seconds

        setLoadingProgress({ stage: stages[0], current: 1, total: stages.length, percent: 8 });
        
        const searchResponse = await axios.post(`${API_BASE_URL}/market-intelligence`, {
          query: query,
          max_results_per_dimension: 8,
          include_ai_insights: true,
          format_type: 'api'
        });
        
        clearInterval(progressInterval);
        setLoadingProgress({ stage: 'Complete!', current: stages.length, total: stages.length, percent: 100 });
        
        setResults({ mode: 'new', data: searchResponse.data.data });
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during search');
    } finally {
      setLoading(false);
      // Reset progress after a short delay
      setTimeout(() => setLoadingProgress({ stage: '', current: 0, total: 0, percent: 0 }), 1000);
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

            <Box sx={{ display: 'flex', gap: 3, alignItems: 'center', flexWrap: 'wrap' }}>
              {/* Search Mode Toggle */}
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant={searchMode === 'old' ? 'contained' : 'outlined'}
                  onClick={() => setSearchMode('old')}
                  disabled={loading}
                  size="small"
                >
                  Old Search (Portfolio Focus)
                </Button>
                <Button
                  variant={searchMode === 'new' ? 'contained' : 'outlined'}
                  onClick={() => setSearchMode('new')}
                  disabled={loading}
                  size="small"
                  color="secondary"
                >
                  New Search (Market Intelligence)
                </Button>
              </Box>

              {searchMode === 'old' && (
              <FormControlLabel
                control={
                  <Switch
                    checked={includeWeb}
                    onChange={(e) => setIncludeWeb(e.target.checked)}
                    disabled={loading}
                  />
                }
                label="Include Live Web Results"
              />
              )}

              {searchMode === 'new' && (
                <Chip 
                  label="Interactive: Click companies for deep dive" 
                  color="secondary" 
                  variant="outlined"
                />
              )}
            </Box>
          </Paper>

          {/* Loading Progress Bar */}
          {loading && loadingProgress.stage && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <Box sx={{ width: '100%' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    {loadingProgress.stage}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {loadingProgress.current} / {loadingProgress.total}
                  </Typography>
                </Box>
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

          {results && results.mode === 'old' && (
            <Box>
              {/* OLD SEARCH RESULTS */}
              {results.data.insights && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                    INSIGHTS (Portfolio Focus)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {results.data.insights.summary && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="body1" component="div">
                        <div dangerouslySetInnerHTML={{
                          __html: results.data.insights.summary
                            .replace(/^(#+)\s*(.*?)(<br>|\n|$)/gm, '<strong>$2</strong><br>')
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                            .replace(/\n/g, '<br>')
                        }} />
                      </Typography>
                    </Box>
                  )}
                </Paper>
              )}

              {/* Database Results */}
              {results.data.database && results.data.database.length > 0 && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    FROM YOUR CURATED DATABASE ({results.data.database.length} results)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {(() => {
                    // Separate Product Hunt products and articles
                    const phProducts = results.data.database.filter(item => item.source_type === 'product_hunt');
                    const articles = results.data.database.filter(item => item.source_type !== 'product_hunt');
                    
                    // Combine with Product Hunt products first
                    const sortedResults = [...phProducts, ...articles];
                    
                    return sortedResults.map((item, index) => (
                      <Card key={index} sx={{ mb: 2 }}>
                        <CardContent>
                          {/* Handle Product Hunt products differently */}
                          {item.source_type === 'product_hunt' ? (
                            <Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6" sx={{ mr: 1 }}>🚀</Typography>
                                <Typography variant="h6" gutterBottom>{item.product_name}</Typography>
                                <Chip 
                                  label="PH" 
                                  size="small" 
                                  color="secondary" 
                                  sx={{ ml: 1 }}
                                />
                              </Box>
                              {item.overview && (
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                  {item.overview}
                                </Typography>
                              )}
                              <Box sx={{ mt: 1 }}>
                                {item.product_link && (
                                  <Typography variant="body2" sx={{ mb: 0.5 }}>
                                    <strong>Product Hunt:</strong> <a href={item.product_link} target="_blank" rel="noopener noreferrer">{item.product_link}</a>
                                  </Typography>
                                )}
                                {item.weighted_score && (
                                  <Typography variant="body2" color="primary">
                                    <strong>Relevance:</strong> {(item.weighted_score * 100).toFixed(1)}% (boosted)
                                  </Typography>
                                )}
                              </Box>
                            </Box>
                          ) : (
                            <Box>
                              <Typography variant="h6" gutterBottom>📰 {item.title}</Typography>
                              <Typography variant="body2" color="text.secondary" gutterBottom>{item.url}</Typography>
                              {item.company_names && (
                                <Box sx={{ mt: 1 }}>
                                  <Typography variant="body2"><strong>Companies:</strong> {item.company_names.join(', ')}</Typography>
                                </Box>
                              )}
                              {item.similarity && (
                                <Typography variant="body2" color="primary" sx={{ mt: 1 }}>
                                  <strong>Relevance:</strong> {(item.similarity * 100).toFixed(1)}%
                                </Typography>
                              )}
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    ));
                  })()}
                </Paper>
              )}

              {/* Web Results */}
              {results.data.web && results.data.web.length > 0 && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    FROM LIVE WEB ({results.data.web.length} results)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {results.data.web.map((item, index) => (
                    <Card key={index} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>{item.title}</Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>{item.description}</Typography>
                        <Typography variant="body2" color="text.secondary">{item.url}</Typography>
                      </CardContent>
                    </Card>
                  ))}
                </Paper>
              )}
            </Box>
          )}

          {results && results.mode === 'new' && (
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
                <Typography variant="body1" sx={{ mb: 3 }}>
                  {(() => {
                    let marketState = results.data.executive_summary || "Analysis of current market conditions and trends.";

                    // Add specific insights from dimensions if available
                    if (results.data.dimensions) {
                      const insights = [];

                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.key_findings && dimension.key_findings.length > 0) {
                          // Look for specific market insights
                          dimension.key_findings.forEach(finding => {
                            if (finding.toLowerCase().includes('market') ||
                                finding.toLowerCase().includes('trend') ||
                                finding.toLowerCase().includes('growth') ||
                                finding.toLowerCase().includes('adoption')) {
                              insights.push(finding);
                            }
                          });
                        }
                      });

                      if (insights.length > 0) {
                        marketState += " Key market insights include: " + insights.slice(0, 2).join("; ");
                      }
                    }

                    return marketState;
                  })()}
                </Typography>

                {/* Discovered Technologies & Innovations */}
                <Typography variant="h6" gutterBottom>
                  Discovered Technologies & Innovations
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
                              technologyDetails[tech] = [];
                            }
                            technologyDetails[tech] = technologyDetails[tech].concat(usages);
                          });
                        }
                      });
                    }

                    const technologiesWithDetails = Object.keys(technologyDetails);
                    // Limit to top 12 technologies
                    const limitedTechnologies = technologiesWithDetails.slice(0, 12);

                    return limitedTechnologies.length > 0 ? (
                      <Grid container spacing={2}>
                        {limitedTechnologies.map((tech, index) => (
                          <Grid item xs={12} sm={6} md={4} key={index}>
                            <Paper sx={{ p: 2, height: '100%' }}>
                              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                                {tech}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                {technologyDetails[tech][0] || `Technology used in financial data analysis and automation`}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    ) : (
                      <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          No specific innovative technologies discovered in the current search. This could mean:
                        </Typography>
                        <Box sx={{ pl: 2 }}>
                          <Typography variant="body2" sx={{ mb: 1 }}>• The search results may not contain specific technology details</Typography>
                          <Typography variant="body2" sx={{ mb: 1 }}>• Companies may be using traditional approaches</Typography>
                          <Typography variant="body2" sx={{ mb: 1 }}>• Try refining your search query for more specific results</Typography>
                          <Typography variant="body2">• Consider searching for specific technology terms</Typography>
                        </Box>
                      </Box>
                    );
                  })()}
                </Box>

                {/* Discovered Business Models */}
                <Typography variant="h6" gutterBottom>
                  Discovered Business Models & Innovations
                </Typography>
                <Box sx={{ mb: 3 }}>
                  {(() => {
                    // Extract discovered business models from dimensions
                    const discoveredBusinessModels = [];
                    const businessModelCompanies = {};

                    if (results.data.dimensions) {
                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.business_model_details && dimension.business_model_details.length > 0) {
                          dimension.business_model_details.forEach(model => {
                            if (!discoveredBusinessModels.includes(model)) {
                              discoveredBusinessModels.push(model);
                            }
                          });
                        }
                      });
                    }

                    // Extract companies from Product Hunt products
                    if (results.data.dimensions) {
                      Object.values(results.data.dimensions).forEach(dimension => {
                        if (dimension.articles) {
                          dimension.articles.forEach(article => {
                            if (article.source === 'product_hunt' && article.Business) {
                              const businessModel = article.Business;
                              const companyName = article.title;
                              
                              if (!businessModelCompanies[businessModel]) {
                                businessModelCompanies[businessModel] = [];
                              }
                              if (!businessModelCompanies[businessModel].includes(companyName)) {
                                businessModelCompanies[businessModel].push(companyName);
                              }
                            }
                          });
                        }
                      });
                    }

                    // Limit to top 12 business models
                    const limitedBusinessModels = discoveredBusinessModels.slice(0, 12);

                    return (
                      <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                          Discovered business models and innovative approaches in the market:
                        </Typography>

                        {limitedBusinessModels.length > 0 ? (
                          <Grid container spacing={2}>
                            {limitedBusinessModels.map((model, index) => {
                              // Remove innovation labels from the model text
                              let cleanModel = model.replace(/\s*\(Innovation:.*?\)\s*/g, '');

                              return (
                                <Grid item xs={12} md={6} key={index}>
                                  <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                                      {cleanModel.split(':')[0] || 'Business Model'}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      {cleanModel.split(':')[1] || cleanModel}
                                    </Typography>
                                  </Paper>
                                </Grid>
                              );
                            })}
                          </Grid>
                        ) : (
                          <Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              No specific business model innovations discovered yet. The market appears to be using traditional approaches:
                            </Typography>
                            <Grid container spacing={2}>
                              {Object.entries(businessModelCompanies).map(([model, companies], index) => (
                                <Grid item xs={12} md={6} key={index}>
                                  <Paper sx={{ p: 2, bgcolor: 'primary.50' }}>
                                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                                      {model}
                                    </Typography>
                                    <Typography variant="body2" sx={{ mb: 2 }}>
                                      Traditional business model with {companies.length} companies identified
                                    </Typography>
                                    {companies.length > 0 && (
                                      <Box>
                                        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>Companies:</Typography>
                                        {companies.slice(0, 3).map((company, idx) => (
                                          <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>
                                            • {company}
                                          </Typography>
                                        ))}
                                        {companies.length > 3 && (
                                          <Typography variant="body2" color="text.secondary">
                                            ... and {companies.length - 3} more
                                          </Typography>
                                        )}
                                      </Box>
                                    )}
                                  </Paper>
                                </Grid>
                              ))}
                            </Grid>
                          </Box>
                        )}
                      </Box>
                    );
                  })()}
                </Box>
              </Paper>

              {/* Companies Section - Product Hunt first, then regular companies */}
              {(() => {
                // Extract Product Hunt products from all dimension results
                const phProducts = [];
                const regularCompanies = [];

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
                            type: 'product_hunt'
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

                return (uniquePhProducts.length > 0 || regularCompanies.length > 0) && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                      Companies & Products
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                    {/* Product Hunt Products First */}
                    {uniquePhProducts.length > 0 && (
                      <Box sx={{ mb: 3 }}>
                       
                        <Grid container spacing={2}>
                          {uniquePhProducts.map((product, index) => (
                            <Grid item xs={12} md={6} key={`ph-${index}`}>
                              <Card
                                sx={{
                                  height: '100%',
                                  cursor: product.url ? 'pointer' : 'default',
                                  '&:hover': product.url ? {
                                    boxShadow: 3,
                                    transform: 'translateY(-2px)',
                                    transition: 'all 0.2s ease-in-out'
                                  } : {}
                                }}
                                onClick={() => product.url && window.open(product.url, '_blank')}
                              >
                                <CardContent>
                                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                    <Typography variant="h6" sx={{ flex: 1 }}>{product.name}</Typography>
                                    <Chip label="PH" size="small" color="secondary" />
                                  </Box>

                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    {product.description || "Emerging product in the market"}
                                  </Typography>

                                  {/* Business Model Description */}
                                  {product.Business && (
                                    <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                        💼 Business Model: {product.Business}
                                      </Typography>
                                      <Typography variant="body2" color="text.secondary">
                                        {product.Business} - Discovered innovative business model in the market. This represents a new approach to monetization and value delivery in the space.
                                      </Typography>
                                    </Box>
                                  )}

                                  {/* Competitive Moat */}
                                  {product.Moat && (
                                    <Box sx={{ mb: 2, p: 2, bgcolor: 'blue.50', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                        🏰 Competitive Moat
                                      </Typography>
                                      <Typography variant="body2" color="text.secondary">
                                        {product.Moat}
                                      </Typography>
                                    </Box>
                                  )}

                                  <Box sx={{ mt: 1 }}>
                                    {product.product_link && (
                                      <Typography variant="body2" sx={{ mb: 0.5 }}>
                                        <strong>Product Hunt:</strong> <a href={product.product_link} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>{product.product_link}</a>
                                      </Typography>
                                    )}
                                  </Box>
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

