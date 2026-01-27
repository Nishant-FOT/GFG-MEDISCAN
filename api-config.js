// API Configuration for MediScan
// Replace with your actual Hugging Face Space URL after deployment
const API_BASE_URL = 'https://huggingface.co/spaces/NishantFOT/MediScanS/api';

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_BASE_URL };
}
