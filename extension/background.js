// Default settings
const DEFAULT_SETTINGS = {
  highlightColor: '#2563eb',
  animationSpeed: 'normal',
  notifications: false,
  scanSpeed: 3
};

const DEFAULT_STATS = {
  scanned: 0,
  flagged: 0,
  startTime: Date.now(),
  accuracy: 0,
  rate: 0,
  lastReset: Date.now()
};

// Initialize extension state
chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed/updated');
  
  // Initialize with default values if not already set
  chrome.storage.local.get(['settings', 'stats', 'isEnabled', 'highlightedTexts'], (data) => {
    const newData = {};
    
    if (!data.settings) {
      console.log('Initializing default settings');
      newData.settings = DEFAULT_SETTINGS;
    }
    
    if (!data.stats) {
      console.log('Initializing default stats');
      newData.stats = DEFAULT_STATS;
    }
    
    if (data.isEnabled === undefined) {
      console.log('Initializing enabled state');
      newData.isEnabled = false;
    }
    
    if (!data.highlightedTexts) {
      console.log('Initializing highlighted texts array');
      newData.highlightedTexts = [];
    }

    if (Object.keys(newData).length > 0) {
      chrome.storage.local.set(newData, () => {
        console.log('Initial state set:', newData);
      });
    }
  });
});

// Handle messages from content script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Received message:', message);
  
  if (message.action === 'showNotification' && message.text) {
    // Create notification
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'images/icon128.png',
      title: 'Review Filter',
      message: message.text
    });
  }
  
  // Update badge when new highlights are found
  if (message.action === 'newHighlight') {
    chrome.storage.local.get(['highlightedTexts'], ({ highlightedTexts = [] }) => {
      const count = highlightedTexts.length + 1;
      chrome.action.setBadgeText({ text: count.toString() });
      chrome.action.setBadgeBackgroundColor({ color: '#2563eb' });
    });
  }
});

// Reset stats daily
setInterval(() => {
  chrome.storage.local.get(['stats'], ({ stats }) => {
    const now = Date.now();
    const dayInMs = 24 * 60 * 60 * 1000;
    
    if (stats && (now - stats.lastReset > dayInMs)) {
      console.log('Resetting daily stats');
      const newStats = {
        ...DEFAULT_STATS,
        lastReset: now
      };
      chrome.storage.local.set({ stats: newStats });
    }
  });
}, 60 * 60 * 1000); // Check every hour

// Update badge when extension is first loaded
chrome.storage.local.get(['highlightedTexts'], ({ highlightedTexts = [] }) => {
  if (highlightedTexts.length > 0) {
    chrome.action.setBadgeText({ text: highlightedTexts.length.toString() });
    chrome.action.setBadgeBackgroundColor({ color: '#2563eb' });
  }
});