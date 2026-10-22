import axios from 'axios';
// The production Express app serves the UI and API from one origin. A relative
// URL also avoids localhost/127.0.0.1 CORS mismatches in the browser.
export const api=axios.create({baseURL:import.meta.env.VITE_API_URL||'/api'});
