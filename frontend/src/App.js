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
  Divider
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
  // Remove limit state and field
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  // Query type detection happens in backend but not displayed in UI

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Perform search (query type detection happens in backend for logging)
      const searchResponse = await axios.post(`${API_BASE_URL}/search`, {
        query: query,
        include_web: includeWeb,
        limit: 7,
        generate_insights: true
      });

      setResults(searchResponse.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during search');
    } finally {
      setLoading(false);
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
              VC-Focused Search System
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Intelligent search with portfolio company context and competitive intelligence
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search for industry trends, company news, or portfolio updates..."
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

            <Box sx={{ display: 'flex', gap: 3, alignItems: 'center' }}>
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
              {/* Remove the TextField component for limit */}
            </Box>
          </Paper>

          {/* Query analysis removed from UI - kept for backend logging only */}

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {results && (
            <Box>
              {/* AI Insights Section */}
              {results.insights && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h5" gutterBottom>
                    VC ACTIONABLE INSIGHTS
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                  {results.insights.summary && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="body1" component="div">
                        <div dangerouslySetInnerHTML={{
                          __html: results.insights.summary
                            // Convert markdown headings to bold and remove hashtags
                            .replace(/^(#+)\s*(.*?)(<br>|\n|$)/gm, '<strong>$2</strong><br>')
                            // Convert **bold** markdown to <strong>
                            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                            // Convert remaining newlines to <br>
                            .replace(/\n/g, '<br>')
                        }} />
                      </Typography>
                    </Box>
                  )}
                </Paper>
              )}

              {/* Database Results */}
              {results.database && results.database.length > 0 && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    FROM YOUR CURATED DATABASE ({results.database.length} results)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                  {results.database.map((item, index) => (
                    <Card key={index} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          {item.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {item.url}
                        </Typography>
                        {item.company_names && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              <strong>Companies:</strong> {item.company_names.join(', ')}
                            </Typography>
                          </Box>
                        )}
                        {item.published_date && (
                          <Typography variant="body2" color="text.secondary">
                            Published: {item.published_date}
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </Paper>
              )}

              {/* Web Results */}
              {results.web && results.web.length > 0 && (
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    FROM LIVE WEB ({results.web.length} results)
                  </Typography>
                  <Divider sx={{ mb: 2 }} />

                  {results.web.map((item, index) => (
                    <Card key={index} sx={{ mb: 2 }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          {item.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {item.description}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {item.url}
                        </Typography>
                        {item.published_date && (
                          <Typography variant="body2" color="text.secondary">
                            Published: {item.published_date}
                          </Typography>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </Paper>
              )}

              {(!results.database || results.database.length === 0) &&
               (!results.web || results.web.length === 0) && (
                <Alert severity="info">
                  No results found for this query. Try adjusting your search terms or including web results.
                </Alert>
              )}
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;

