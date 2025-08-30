// Enhanced Pipeline Processor Extension JavaScript

const API_BASE_URL = 'http://localhost:8000';

class PipelineProcessor {
  constructor() {
    this.isProcessing = false;
    this.currentMode = 'manual';
    this.extractedData = null;
    this.init();
  }

  init() {
    this.bindEvents();
    this.checkAPIHealth();
    this.loadSettings();
  }

  bindEvents() {
    // Mode toggle
    document.getElementById('formModeBtn').addEventListener('click', () => this.switchMode('manual'));
    document.getElementById('autoModeBtn').addEventListener('click', () => this.switchMode('auto'));

    // Optional fields toggle
    document.getElementById('optionalToggle').addEventListener('click', this.toggleOptionalFields);

    // Form submission
    document.getElementById('reviewForm').addEventListener('submit', (e) => this.handleFormSubmit(e));

    // Auto extract
    document.getElementById('extractBtn').addEventListener('click', () => this.extractFromPage());
    document.getElementById('useExtractedBtn').addEventListener('click', () => this.useExtractedData());

    // Results navigation
    document.getElementById('backBtn').addEventListener('click', () => this.showMainContent());

    // Footer buttons
    document.getElementById('settingsBtn').addEventListener('click', () => this.showSettings());
    document.getElementById('helpBtn').addEventListener('click', () => this.showHelp());
  }

  async checkAPIHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      
      if (data.status === 'healthy') {
        this.updateStatus('Connected', 'success');
      } else {
        this.updateStatus('API Unhealthy', 'warning');
      }
    } catch (error) {
      this.updateStatus('Disconnected', 'error');
      console.error('API health check failed:', error);
    }
  }

  updateStatus(text, type = 'success') {
    const statusText = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');
    
    statusText.textContent = text;
    
    // Remove all status classes
    statusDot.classList.remove('status-success', 'status-warning', 'status-error');
    
    // Add appropriate class
    if (type === 'success') {
      statusDot.style.background = '#22c55e';
    } else if (type === 'warning') {
      statusDot.style.background = '#f59e0b';
    } else if (type === 'error') {
      statusDot.style.background = '#ef4444';
    }
  }

  switchMode(mode) {
    this.currentMode = mode;
    
    // Update button states
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    
    if (mode === 'manual') {
      document.getElementById('formModeBtn').classList.add('active');
      document.getElementById('manualInputSection').classList.remove('hidden');
      document.getElementById('autoExtractSection').classList.add('hidden');
    } else {
      document.getElementById('autoModeBtn').classList.add('active');
      document.getElementById('manualInputSection').classList.add('hidden');
      document.getElementById('autoExtractSection').classList.remove('hidden');
    }
  }

  toggleOptionalFields() {
    const toggle = document.getElementById('optionalToggle');
    const fields = document.getElementById('optionalFields');
    const icon = toggle.querySelector('i');
    
    const isExpanded = fields.classList.contains('expanded');
    
    if (isExpanded) {
      fields.classList.remove('expanded');
      toggle.classList.remove('expanded');
    } else {
      fields.classList.add('expanded');
      toggle.classList.add('expanded');
    }
  }

  async handleFormSubmit(e) {
    e.preventDefault();
    
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    this.showLoading();
    
    try {
      // Collect form data
      const formData = new FormData(e.target);
      const reviewData = {};
      
      // Convert FormData to object
      for (let [key, value] = formData) {
        if (value.trim() !== '') {
          // Convert numeric fields
          if (['rating', 'avg_rating', 'num_of_reviews', 'latitude', 'longitude'].includes(key)) {
            reviewData[key] = parseFloat(value);
          } else {
            reviewData[key] = value;
          }
        }
      }
      
      // Add timestamp if not provided
      if (!reviewData.time) {
        reviewData.time = Date.now();
      }
      
      console.log('Submitting review data:', reviewData);
      
      // Process through pipeline
      const result = await this.processReview(reviewData);
      
      this.hideLoading();
      this.showResults(result);
      
    } catch (error) {
      this.hideLoading();
      this.showError('Processing failed: ' + error.message);
      console.error('Form submission error:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  async processReview(reviewData) {
    const stages = ['Stage 1: Data Processing', 'Stage 2: Rule Filtering', 'Stage 3: AI Model', 'Stage 4: Relevance Check', 'Stage 5: Evaluation'];
    let currentStage = 0;
    
    // Simulate stage progress
    const progressInterval = setInterval(() => {
      if (currentStage < stages.length) {
        this.updateLoadingStatus(stages[currentStage], ((currentStage + 1) / stages.length) * 100);
        currentStage++;
      }
    }, 800);
    
    try {
      const response = await fetch(`${API_BASE_URL}/process_review`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reviewData),
      });
      
      clearInterval(progressInterval);
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const result = await response.json();
      return result;
      
    } catch (error) {
      clearInterval(progressInterval);
      throw error;
    }
  }

  async extractFromPage() {
    this.updateStatus('Extracting...', 'warning');
    
    try {
      // Send message to content script to extract review data
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      const response = await chrome.tabs.sendMessage(tab.id, {
        action: 'extractReviewData'
      });
      
      if (response && response.success) {
        this.extractedData = response.data;
        this.showExtractedData(response.data);
        this.updateStatus('Extracted', 'success');
      } else {
        throw new Error('No review data found on this page');
      }
      
    } catch (error) {
      this.showError('Extraction failed: ' + error.message);
      this.updateStatus('Extract Failed', 'error');
      console.error('Extraction error:', error);
    }
  }

  showExtractedData(data) {
    const container = document.getElementById('extractedData');
    const content = document.getElementById('extractedDataContent');
    
    content.textContent = JSON.stringify(data, null, 2);
    container.style.display = 'block';
  }

  useExtractedData() {
    if (!this.extractedData) return;
    
    // Switch to manual mode and populate form
    this.switchMode('manual');
    
    // Populate form fields
    Object.keys(this.extractedData).forEach(key => {
      const input = document.querySelector(`[name="${key}"]`);
      if (input && this.extractedData[key]) {
        input.value = this.extractedData[key];
      }
    });
    
    // Show success message
    this.showNotification('Form populated with extracted data!');
  }

  showLoading() {
    document.getElementById('loadingOverlay').classList.remove('hidden');
    this.updateLoadingStatus('Initializing pipeline...', 0);
  }

  hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
  }

  updateLoadingStatus(status, progress) {
    document.getElementById('loadingStatus').textContent = status;
    document.getElementById('progressFill').style.width = `${progress}%`;
  }

  showResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    
    // Hide main content, show results
    document.querySelector('.main-content').classList.add('hidden');
    resultsSection.classList.remove('hidden');
    
    // Populate results
    resultsContent.innerHTML = this.generateResultsHTML(result);
  }

  generateResultsHTML(result) {
    if (!result.success) {
      return `
        <div class="error-result">
          <i class="ri-error-warning-line"></i>
          <h3>Processing Failed</h3>
          <p>${result.error_message || 'Unknown error occurred'}</p>
          <div class="processing-info">
            <small>Processing ID: ${result.processing_id}</small>
            <small>Time: ${result.processing_time?.toFixed(2)}s</small>
          </div>
        </div>
      `;
    }
    
    const predictions = result.final_predictions;
    const stages = result.stage_results;
    
    return `
      <div class="success-result">
        <div class="result-header">
          <i class="ri-check-circle-line"></i>
          <h3>Processing Complete</h3>
          <div class="processing-info">
            <span>ID: ${result.processing_id}</span>
            <span>Time: ${result.processing_time?.toFixed(2)}s</span>
          </div>
        </div>
        
        <div class="predictions-section">
          <h4>Final Predictions</h4>
          <div class="prediction-cards">
            ${predictions.model_prediction !== undefined ? `
              <div class="prediction-card">
                <div class="prediction-label">AI Validity</div>
                <div class="prediction-value ${predictions.model_prediction === 0 ? 'valid' : 'invalid'}">
                  ${predictions.model_prediction === 0 ? 'Valid' : 'Invalid'}
                </div>
                <div class="prediction-confidence">
                  Confidence: ${(predictions.model_confidence * 100).toFixed(1)}%
                </div>
              </div>
            ` : ''}
            
            ${predictions.relevance_score !== undefined ? `
              <div class="prediction-card">
                <div class="prediction-label">Relevance</div>
                <div class="prediction-value ${predictions.is_relevant === 1 ? 'relevant' : 'irrelevant'}">
                  ${predictions.is_relevant === 1 ? 'Relevant' : 'Not Relevant'}
                </div>
                <div class="prediction-confidence">
                  Score: ${(predictions.relevance_score * 100).toFixed(1)}%
                </div>
              </div>
            ` : ''}
          </div>
        </div>
        
        <div class="stages-section">
          <h4>Pipeline Stages</h4>
          <div class="stages-list">
            ${Object.keys(stages).map(stageName => `
              <div class="stage-item">
                <div class="stage-name">${stageName}</div>
                <div class="stage-info">
                  ${stages[stageName].row_count} rows processed
                </div>
              </div>
            `).join('')}
          </div>
        </div>
        
        <div class="features-section">
          <h4>Generated Features</h4>
          <div class="features-grid">
            ${Object.keys(predictions).filter(key => 
              ['text_length', 'word_count', 'hour_of_day', 'is_weekend', 'rating_deviation', 'has_business_response'].includes(key)
            ).map(key => `
              <div class="feature-item">
                <span class="feature-name">${key.replace(/_/g, ' ')}</span>
                <span class="feature-value">${predictions[key]}</span>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  showMainContent() {
    document.querySelector('.main-content').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
  }

  showError(message) {
    this.showNotification(message, 'error');
  }

  showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style notification
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 12px 16px;
      background: ${type === 'error' ? '#ef4444' : '#22c55e'};
      color: white;
      border-radius: 8px;
      font-size: 14px;
      z-index: 10001;
      transform: translateX(100%);
      transition: transform 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
      notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
      notification.style.transform = 'translateX(100%)';
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  }

  showSettings() {
    console.log('Settings clicked');
    // TODO: Implement settings modal
  }

  showHelp() {
    console.log('Help clicked');
    // TODO: Implement help modal
  }

  loadSettings() {
    // Load saved settings
    chrome.storage.local.get(['pipelineSettings'], ({ pipelineSettings }) => {
      if (pipelineSettings) {
        // Apply saved settings
        console.log('Loaded settings:', pipelineSettings);
      }
    });
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new PipelineProcessor();
});

// Add CSS for dynamic elements
const style = document.createElement('style');
style.textContent = `
  .error-result, .success-result {
    padding: 20px;
  }
  
  .result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .result-header i {
    font-size: 24px;
    color: #22c55e;
  }
  
  .error-result .result-header i {
    color: #ef4444;
  }
  
  .processing-info {
    margin-left: auto;
    text-align: right;
    font-size: 12px;
    color: #6b7280;
  }
  
  .processing-info span {
    display: block;
  }
  
  .predictions-section, .stages-section, .features-section {
    margin-bottom: 24px;
  }
  
  .predictions-section h4, .stages-section h4, .features-section h4 {
    margin-bottom: 12px;
    color: #374151;
    font-size: 16px;
  }
  
  .prediction-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  
  .prediction-card {
    padding: 16px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    text-align: center;
  }
  
  .prediction-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 8px;
    text-transform: uppercase;
    font-weight: 500;
  }
  
  .prediction-value {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
  }
  
  .prediction-value.valid, .prediction-value.relevant {
    color: #22c55e;
  }
  
  .prediction-value.invalid, .prediction-value.irrelevant {
    color: #ef4444;
  }
  
  .prediction-confidence {
    font-size: 12px;
    color: #6b7280;
  }
  
  .stages-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  
  .stage-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    background: #f8fafc;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
  }
  
  .stage-name {
    font-weight: 500;
    color: #374151;
  }
  
  .stage-info {
    font-size: 12px;
    color: #6b7280;
  }
  
  .features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  
  .feature-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 4px;
    font-size: 12px;
  }
  
  .feature-name {
    color: #6b7280;
    text-transform: capitalize;
  }
  
  .feature-value {
    color: #374151;
    font-weight: 500;
  }
`;
document.head.appendChild(style);
