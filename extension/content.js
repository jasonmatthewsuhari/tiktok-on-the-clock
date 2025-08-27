console.log('Content script loaded and initialized');

let isEnabled = false;
let highlightInterval;
let highlightCount = 0;

const DEFAULT_SETTINGS = {
  highlightColor: '#2563eb',
  animationSpeed: 'normal',
  notifications: false,
  scanSpeed: 3
};

let settings = { ...DEFAULT_SETTINGS };

// Animation durations based on speed setting
const ANIMATION_SPEEDS = {
  fast: 1500,
  normal: 2500,
  slow: 3500
};

// Scan intervals based on speed slider (1-5)
const SCAN_INTERVALS = {
  1: 5000,  // 5 seconds
  2: 4000,  // 4 seconds
  3: 3000,  // 3 seconds
  4: 2000,  // 2 seconds
  5: 1000   // 1 second
};

// Function to check if an element is visible
function isElementVisible(element) {
  const rect = element.getBoundingClientRect();
  const isInViewport = (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
  
  const style = window.getComputedStyle(element);
  const isVisibleStyle = style.display !== 'none' && 
                        style.visibility !== 'hidden' && 
                        style.opacity !== '0';
  
  return isInViewport && isVisibleStyle;
}

// Function to get random text elements from the page
function getRandomTextElement() {
  console.log('Searching for text elements...');
  
  // Get all text elements
  const textElements = Array.from(document.querySelectorAll('p, span, div, td, li, h1, h2, h3, h4, h5, h6'))
    .filter(el => {
      const text = el.textContent.trim();
      const hasValidText = text.length > 10 && text.length < 1000;
      const isVisible = isElementVisible(el);
      const notHighlighted = !el.hasAttribute('data-highlighted');
      return hasValidText && isVisible && notHighlighted;
    });
  
  console.log(`Found ${textElements.length} suitable text elements`);
  
  if (textElements.length === 0) {
    console.log('No suitable elements found, refreshing previously highlighted elements');
    // Reset highlighted elements after all have been processed
    document.querySelectorAll('[data-highlighted]').forEach(el => {
      el.removeAttribute('data-highlighted');
    });
    return null;
  }
  
  const element = textElements[Math.floor(Math.random() * textElements.length)];
  element.setAttribute('data-highlighted', 'true');
  console.log('Selected element:', element.textContent.trim().substring(0, 50) + '...');
  return element;
}

// Create a visual indicator for the extension status
const statusIndicator = document.createElement('div');
statusIndicator.style.cssText = `
  position: fixed;
  bottom: 20px;
  right: 20px;
  padding: 8px 16px;
  background: rgba(37, 99, 235, 0.9);
  color: white;
  border-radius: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 14px;
  z-index: 10000;
  display: none;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
  backdrop-filter: blur(4px);
`;

const statusIcon = document.createElement('div');
statusIcon.style.cssText = `
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #22c55e;
`;

statusIndicator.appendChild(statusIcon);
statusIndicator.appendChild(document.createTextNode('Scanning...'));
document.body.appendChild(statusIndicator);

// Function to highlight an element
function highlightElement(element) {
  if (!element) {
    console.log('No element to highlight');
    return;
  }
  
  console.log('Highlighting element:', element.textContent.trim().substring(0, 50) + '...');
  highlightCount++;
  
  // Update status indicator
  statusIndicator.lastChild.textContent = `Scanning... (${highlightCount} found)`;
  
  // Create highlight overlay
  const highlight = document.createElement('div');
  highlight.style.cssText = `
    position: absolute;
    background-color: ${settings.highlightColor}1a;
    border: 2px solid ${settings.highlightColor};
    border-radius: 8px;
    pointer-events: none;
    z-index: 9999;
    animation: highlightFade ${ANIMATION_SPEEDS[settings.animationSpeed]}ms cubic-bezier(0.4, 0, 0.2, 1) forwards;
    backdrop-filter: blur(2px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
  `;

  // Position the highlight
  const rect = element.getBoundingClientRect();
  highlight.style.top = `${rect.top + window.scrollY}px`;
  highlight.style.left = `${rect.left + window.scrollX}px`;
  highlight.style.width = `${rect.width}px`;
  highlight.style.height = `${rect.height}px`;

  document.body.appendChild(highlight);

  // Send highlighted text to popup
  chrome.runtime.sendMessage({
    action: 'newHighlight',
    text: element.textContent.trim()
  });

  // Show notification if enabled
  if (settings.notifications) {
    chrome.runtime.sendMessage({
      action: 'showNotification',
      text: 'New text highlighted!'
    });
  }

  // Remove highlight after animation
  setTimeout(() => {
    highlight.remove();
  }, ANIMATION_SPEEDS[settings.animationSpeed]);
}

// Add necessary styles
const style = document.createElement('style');
style.textContent = `
  @keyframes highlightFade {
    0% { 
      opacity: 0; 
      transform: scale(0.98); 
      box-shadow: 0 0 0 rgba(37, 99, 235, 0);
    }
    20% { 
      opacity: 1; 
      transform: scale(1);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    80% { 
      opacity: 1; 
      transform: scale(1);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    100% { 
      opacity: 0; 
      transform: scale(0.98);
      box-shadow: 0 0 0 rgba(37, 99, 235, 0);
    }
  }

  @media (prefers-color-scheme: dark) {
    .highlight-overlay {
      background-color: rgba(59, 130, 246, 0.15) !important;
      border-color: rgba(59, 130, 246, 0.8) !important;
    }
  }
`;
document.head.appendChild(style);

// Handle scroll events to update highlight positions
let scrollTimeout;
window.addEventListener('scroll', () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(() => {
    if (isEnabled) {
      // Force a new highlight after scrolling
      const element = getRandomTextElement();
      if (element) {
        highlightElement(element);
      }
    }
  }, 500);
}, { passive: true });

// Start/stop highlighting
function toggleHighlighting(enabled, speed = settings.scanSpeed) {
  console.log('Toggle highlighting:', enabled, 'speed:', speed);
  isEnabled = enabled;
  settings.scanSpeed = speed;
  
  if (isEnabled) {
    console.log('Starting highlight interval with speed:', SCAN_INTERVALS[settings.scanSpeed]);
    statusIndicator.style.display = 'flex';
    highlightInterval = setInterval(() => {
      const element = getRandomTextElement();
      // Always increment scan count
      chrome.runtime.sendMessage({
        action: 'scanComplete',
        timestamp: Date.now()
      });
      
      if (element) {
        highlightElement(element);
      }
  } else {
    console.log('Clearing highlight interval');
    statusIndicator.style.display = 'none';
    clearInterval(highlightInterval);
  }
}

// Update settings
function updateSettings(newSettings) {
  console.log('Updating settings:', newSettings);
  settings = { ...settings, ...newSettings };
  
  if (isEnabled) {
    // Restart highlighting with new settings
    clearInterval(highlightInterval);
    toggleHighlighting(true, settings.scanSpeed);
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Received message:', message);
  if (message.action === 'toggleHighlighting') {
    toggleHighlighting(message.isEnabled, message.speed);
  } else if (message.action === 'updateSettings') {
    updateSettings(message.settings);
  } else if (message.action === 'updateSpeed') {
    settings.scanSpeed = message.speed;
    if (isEnabled) {
      clearInterval(highlightInterval);
      toggleHighlighting(true, message.speed);
    }
  }
});

// Check initial state
console.log('Checking initial state...');
chrome.storage.local.get(['isEnabled', 'settings'], ({ isEnabled: enabled, settings: savedSettings }) => {
  console.log('Initial state:', { enabled, savedSettings });
  // Merge saved settings with defaults
  settings = { ...DEFAULT_SETTINGS, ...(savedSettings || {}) };
  
  // Ensure settings are saved with defaults
  chrome.storage.local.set({ settings });
  
  toggleHighlighting(enabled || false);
});