document.addEventListener('DOMContentLoaded', () => {
  const toggleButton = document.getElementById('toggleButton');
  const toggleStatus = document.getElementById('toggleStatus');
  const detectButton = document.getElementById('detectReviews');
  const extractButton = document.getElementById('extractReview');
  const analyzeButton = document.getElementById('analyzePage');
  const clearButton = document.getElementById('clearHighlights');
  const reviewStatus = document.getElementById('reviewStatus');
  const reviewStatusContent = document.getElementById('reviewStatusContent');

  // Load initial state
  chrome.storage.local.get(['isEnabled'], ({ isEnabled }) => {
    updateToggleStatus(isEnabled || false);
  });

  // Toggle button click handler
  toggleButton.addEventListener('click', () => {
    const isEnabled = toggleButton.classList.toggle('active');
    chrome.storage.local.set({ isEnabled });
    updateToggleStatus(isEnabled);
    
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'toggleHighlighting',
        isEnabled
      });
    });
  });

  // Detect reviews button
  detectButton.addEventListener('click', () => {
    showStatus('Detecting reviews...', 'loading');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'detectReviews'
      }, (response) => {
        if (chrome.runtime.lastError) {
          showStatus('Extension not loaded on this page. Try refreshing.', 'error');
          return;
        }
        if (response && response.success) {
          showStatus(`Found ${response.reviewsFound} reviews (${response.highlightedReviews} highlighted)`, 'success');
          
          if (response.reviews.length > 0) {
            const reviewsList = response.reviews.map(r => 
              `<div class="review-item">
                <span class="confidence">${r.confidence}%</span>
                <span class="review-text">${r.text}</span>
              </div>`
            ).join('');
            
            reviewStatusContent.innerHTML += `<div class="reviews-list">${reviewsList}</div>`;
          }
        } else {
          showStatus('No reviews detected on this page', 'error');
        }
      });
    });
  });

  // Extract review data button
  extractButton.addEventListener('click', () => {
    showStatus('Extracting review data...', 'loading');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'extractReviewData'
      }, (response) => {
        if (chrome.runtime.lastError) {
          showStatus('Extension not loaded on this page. Try refreshing.', 'error');
          return;
        }
        if (response && response.success) {
          showStatus(`Extracted ${Object.keys(response.data).length} data fields`, 'success');
          
          // Display extracted data
          const dataList = Object.entries(response.data).map(([key, value]) => 
            `<div class="data-item">
              <span class="data-key">${key}:</span>
              <span class="data-value">${String(value).substring(0, 50)}${String(value).length > 50 ? '...' : ''}</span>
            </div>`
          ).join('');
          
          reviewStatusContent.innerHTML += `<div class="extracted-data">${dataList}</div>`;
          
          // Copy to clipboard
          navigator.clipboard.writeText(JSON.stringify(response.data, null, 2)).then(() => {
            console.log('Data copied to clipboard');
          });
          
        } else {
          showStatus(response?.error || 'Failed to extract review data', 'error');
        }
      });
    });
  });

  // Analyze page button
  analyzeButton.addEventListener('click', () => {
    showStatus('Analyzing page...', 'loading');
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'analyzePageForReviews'
      }, (response) => {
        if (response && response.success) {
          const analysis = response.analysis;
          showStatus(`Analysis complete: ${analysis.reviewsDetected} reviews found`, 'success');
          
          const analysisHTML = `
            <div class="analysis-results">
              <div class="analysis-item">
                <span class="analysis-label">Page:</span>
                <span class="analysis-value">${analysis.pageTitle}</span>
              </div>
              <div class="analysis-item">
                <span class="analysis-label">Text Elements:</span>
                <span class="analysis-value">${analysis.totalTextElements}</span>
              </div>
              <div class="analysis-item">
                <span class="analysis-label">Reviews Found:</span>
                <span class="analysis-value">${analysis.reviewsDetected}</span>
              </div>
              <div class="analysis-item">
                <span class="analysis-label">High Confidence:</span>
                <span class="analysis-value">${analysis.highConfidenceReviews}</span>
              </div>
              <div class="analysis-item">
                <span class="analysis-label">Medium Confidence:</span>
                <span class="analysis-value">${analysis.mediumConfidenceReviews}</span>
              </div>
            </div>
          `;
          
          reviewStatusContent.innerHTML += analysisHTML;
        } else {
          showStatus('Failed to analyze page', 'error');
        }
      });
    });
  });

  // Clear highlights button
  clearButton.addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'clearHighlights'
      }, (response) => {
        if (response && response.success) {
          showStatus('Highlights cleared', 'success');
          setTimeout(() => {
            reviewStatus.style.display = 'none';
          }, 1500);
        }
      });
    });
  });

  function updateToggleStatus(isEnabled) {
    toggleStatus.textContent = isEnabled ? 'Active' : 'Inactive';
    toggleStatus.className = 'power-state ' + (isEnabled ? 'active' : 'inactive');
    toggleButton.classList.toggle('active', isEnabled);
  }

  function showStatus(message, type = 'info') {
    reviewStatus.style.display = 'block';
    reviewStatusContent.innerHTML = `
      <div class="status-message ${type}">
        ${type === 'loading' ? '<i class="ri-loader-4-line rotating"></i>' : ''}
        ${type === 'success' ? '<i class="ri-check-line"></i>' : ''}
        ${type === 'error' ? '<i class="ri-error-warning-line"></i>' : ''}
        <span>${message}</span>
      </div>
    `;
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
      setTimeout(() => {
        const statusMessage = reviewStatusContent.querySelector('.status-message');
        if (statusMessage && statusMessage.classList.contains('success')) {
          statusMessage.style.opacity = '0.5';
        }
      }, 3000);
    }
  }

  // Listen for AI validation results from content script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'aiValidationResult') {
      const result = message.result;
      const reviewText = message.reviewText;
      
      // Show validation result in popup
      const validationHTML = `
        <div class="validation-result">
          <div class="validation-header">
            <i class="ri-robot-line"></i>
            <span>AI Validation Complete</span>
          </div>
          <div class="validation-details">
            <div class="validation-status ${result.is_valid ? 'valid' : 'invalid'}">
              <strong>${result.is_valid ? 'VALID' : 'INVALID'} REVIEW</strong>
              <span class="confidence">${Math.round(result.confidence * 100)}% confidence</span>
            </div>
            <div class="validation-probabilities">
              <div class="prob-item">
                <span class="prob-label">Valid:</span>
                <span class="prob-value">${Math.round(result.probability_valid * 100)}%</span>
              </div>
              <div class="prob-item">
                <span class="prob-label">Invalid:</span>
                <span class="prob-value">${Math.round(result.probability_invalid * 100)}%</span>
              </div>
            </div>
            <div class="validation-meta">
              <div class="meta-item">Processing: ${Math.round(result.processing_time_ms)}ms</div>
              <div class="meta-item">Model: ${result.model_version}</div>
            </div>
            <div class="review-preview">
              "${reviewText}"
            </div>
          </div>
        </div>
      `;
      
      if (reviewStatusContent.innerHTML.includes('status-message')) {
        reviewStatusContent.innerHTML += validationHTML;
      } else {
        reviewStatusContent.innerHTML = validationHTML;
      }
      
      reviewStatus.style.display = 'block';
    }
  });
});