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
                  {results.data.database.map((item, index) => (
                    <Card key={index} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>{item.title}</Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>{item.url}</Typography>
                        {item.company_names && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2"><strong>Companies:</strong> {item.company_names.join(', ')}</Typography>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  ))}
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
              {/* NEW SEARCH RESULTS - Market Intelligence */}
              {results.data.executive_summary && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                    Market Overview
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    {results.data.executive_summary}
                  </Typography>
                </Paper>
              )}

              {/* Interactive Companies */}
              {results.data.interactive_elements?.companies && results.data.interactive_elements.companies.length > 0 && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                    Companies Found
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Grid container spacing={2}>
                    {results.data.interactive_elements.companies.map((company, index) => {
                      const deepDiveAction = company.actions.find(a => a.type === 'deep_dive');
                      const isExpanded = !!expandedCompanies[company.name];
                      const isLoading = !!deepDiveLoading[company.name];
                      
                      return (
                        <Grid item xs={12} key={index}>
                          <Card>
                            <CardContent>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                                <Box sx={{ flex: 1 }}>
                                  <Typography variant="h6" gutterBottom>{company.name}</Typography>
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    {company.context}
                                  </Typography>
                                  <Chip 
                                    label={`Mentioned in: ${company.mentioned_in.join(', ')}`} 
                                    size="small"
                                  />
                                </Box>
                                <Button
                                  variant={isExpanded ? 'outlined' : 'contained'}
                                  color="secondary"
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
                              </Box>

                              {/* Inline Deep Dive Results */}
                              {isExpanded && expandedCompanies[company.name] && (
                                <Box sx={{ mt: 3, pt: 3, borderTop: '1px solid #e0e0e0' }}>
                                  <Typography variant="h6" gutterBottom color="primary">
                                    Deep Dive Analysis
                                  </Typography>
                                  
                                  <Typography variant="body1" sx={{ mb: 2, fontWeight: 500 }}>
                                    {expandedCompanies[company.name].overview}
                                  </Typography>

                                  {expandedCompanies[company.name].key_facts && expandedCompanies[company.name].key_facts.length > 0 && (
                                    <Box sx={{ mb: 2 }}>
                                      <Typography variant="subtitle2" gutterBottom>Key Facts:</Typography>
                                      {expandedCompanies[company.name].key_facts.slice(0, 5).map((fact, idx) => (
                                        <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>• {fact}</Typography>
                                      ))}
                                    </Box>
                                  )}

                                  <Grid container spacing={2}>
                                    {expandedCompanies[company.name].competitors && expandedCompanies[company.name].competitors.length > 0 && (
                                      <Grid item xs={12} md={6}>
                                        <Typography variant="subtitle2" gutterBottom>Competitors:</Typography>
                                        <Typography variant="body2">{expandedCompanies[company.name].competitors.join(', ')}</Typography>
                                      </Grid>
                                    )}

                                    {expandedCompanies[company.name].business_model && (
                                      <Grid item xs={12} md={6}>
                                        <Typography variant="subtitle2" gutterBottom>Business Model:</Typography>
                                        <Typography variant="body2">{expandedCompanies[company.name].business_model}</Typography>
                                      </Grid>
                                    )}

                                    {expandedCompanies[company.name].technologies_used && expandedCompanies[company.name].technologies_used.length > 0 && (
                                      <Grid item xs={12}>
                                        <Typography variant="subtitle2" gutterBottom>Technologies:</Typography>
                                        <Typography variant="body2">{expandedCompanies[company.name].technologies_used.join(', ')}</Typography>
                                      </Grid>
                                    )}
                                  </Grid>

                                  {expandedCompanies[company.name].recent_developments && expandedCompanies[company.name].recent_developments.length > 0 && (
                                    <Box sx={{ mt: 2 }}>
                                      <Typography variant="subtitle2" gutterBottom>Recent Developments:</Typography>
                                      {expandedCompanies[company.name].recent_developments.slice(0, 3).map((dev, idx) => (
                                        <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>
                                          <strong>{dev.date}:</strong> {dev.description}
                                        </Typography>
                                      ))}
                                    </Box>
                                  )}
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      );
                    })}
                  </Grid>
                </Paper>
              )}

              {/* Removed: Key Players, Emerging Trends, and Recommended Next Steps sections */}
            </Box>
          )}

          {/* Deep dive now shown inline in company cards */}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;

