console.log('🚀 CONTENT SCRIPT LOADED AND INITIALIZED 🚀');
console.log('Extension version: 1.1');
console.log('Current URL:', window.location.href);

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
    
    // Start with immediate review detection
    console.log('🔍 Auto-detecting reviews on activation...');
    setTimeout(() => {
      const reviewCount = ReviewHighlighter.highlightDetectedReviews();
      console.log(`🎯 Auto-detected ${reviewCount} reviews on page`);
      
      // Update status indicator
      if (reviewCount > 0) {
        statusIndicator.innerHTML = `
          <div style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e;"></div>
          <span>Found ${reviewCount} reviews</span>
        `;
      }
    }, 1000);
    
    // Continue with regular text highlighting AND periodic review scanning
    highlightInterval = setInterval(() => {
      const element = getRandomTextElement();
      
      // Every 3rd interval, also scan for new reviews
      if (highlightCount % 3 === 0) {
        console.log('🔄 Periodic review scan...');
        const newReviewCount = ReviewHighlighter.highlightDetectedReviews();
        if (newReviewCount > 0) {
          console.log(`🆕 Found ${newReviewCount} new reviews`);
          statusIndicator.innerHTML = `
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #22c55e;"></div>
            <span>Found ${newReviewCount} reviews</span>
          `;
        }
      }
      
      // Always increment scan count
      chrome.runtime.sendMessage({
        action: 'scanComplete',
        timestamp: Date.now()
      });
      
      if (element) {
        highlightElement(element);
      }
    }, SCAN_INTERVALS[settings.scanSpeed]);
  } else {
    console.log('Clearing highlight interval');
    statusIndicator.style.display = 'none';
    clearInterval(highlightInterval);
    
    // Clear review highlights when disabled
    ReviewHighlighter.clearHighlights();
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

// Advanced review detection system that doesn't rely on CSS classes
const ReviewDetector = {
  // Common review indicators for content analysis
  reviewIndicators: {
    rating: [
      /\b[1-5](\.\d)?[\/\s]*(?:out of|\/|\s)?[5-5]?\s*(?:stars?|⭐|★)/i,
      /★{1,5}|⭐{1,5}/,
      /rating\s*:?\s*[1-5](\.\d)?/i,
      /scored?\s*[1-5](\.\d)?/i,
      /\b([1-9]|10)\/10\b/i,  // Catches "10/10", "8/10", etc.
      /\b[1-5](\.\d)?\s*out\s*of\s*[1-5]\b/i  // "4.5 out of 5"
    ],
    temporal: [
      /\b(?:a|an|\d+)\s*(?:day|week|month|year)s?\s*ago\b/i,
      /\b(?:yesterday|today|last\s+(?:week|month|year))\b/i,
      /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d+/i,
      /\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b/,
      /\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b/
    ],
    reviewContent: [
      /\b(?:great|good|bad|terrible|awful|amazing|excellent|poor|disappointing|recommend|satisfied|unsatisfied)\b/i,
      /\b(?:service|staff|food|atmosphere|experience|visit|quality|price|value)\b/i,
      /\b(?:would|will|definitely|never)\s+(?:come back|return|visit again|recommend)\b/i,
      /\b(?:loved|hated|enjoyed|disliked)\s+(?:it|this|the)\b/i
    ],
    reviewMeta: [
      /\b(?:verified|confirmed)\s+(?:purchase|buyer|customer)\b/i,
      /\b(?:helpful|useful)[\s\?]*\b/i,
      /\b\d+\s*(?:people|users?)\s*found\s*(?:this|it)\s*helpful\b/i
    ]
  },

  // Detect review elements using semantic analysis
  detectReviewElements() {
    console.log('🔍 Starting semantic review detection...');
    
    const potentialReviews = [];
    const textElements = this.getAllTextElements();
    
    console.log(`📝 Analyzing ${textElements.length} text elements...`);
    
    textElements.forEach((element, index) => {
      const reviewData = this.analyzeElementForReview(element, index);
      if (reviewData.isReview) {
        potentialReviews.push(reviewData);
      }
    });

    console.log(`✅ Found ${potentialReviews.length} potential reviews`);
    return potentialReviews;
  },

  getAllTextElements() {
    // Get all elements that could contain review text
    const selectors = [
      'div', 'span', 'p', 'section', 'article', 'li', 'td',
      '[role="listitem"]', '[role="article"]', '[role="region"]'
    ];
    
    const elements = [];
    selectors.forEach(selector => {
      const found = document.querySelectorAll(selector);
      elements.push(...Array.from(found));
    });

    // Filter for elements with meaningful text content
    const candidateElements = elements.filter(el => {
      const text = el.textContent?.trim() || '';
      return text.length > 20 && text.length < 5000 && 
             this.isVisible(el) && 
             !this.isNavigationElement(el);
    });

    // Prioritize Google Maps review containers
    const prioritizedElements = [];
    const regularElements = [];

    candidateElements.forEach(el => {
      // Check if this looks like a Google Maps review structure
      if (this.isGoogleMapsReviewContainer(el)) {
        prioritizedElements.push(el);
      } else {
        regularElements.push(el);
      }
    });

    // Return prioritized elements first, then regular ones
    return [...prioritizedElements, ...regularElements];
  },

  isGoogleMapsReviewContainer(element) {
    // Check for Google Maps review container patterns
    const hasReviewStructure = (
      // Has contributor link (review author)
      element.querySelector('a[href*="/contrib/"]') ||
      element.querySelector('a[href*="/reviews"]') ||
      // Has data attributes typical of review containers
      element.hasAttribute('data-id') ||
      element.hasAttribute('jsdata') ||
      // Contains "More" expand button
      element.querySelector('a[aria-label*="more"]') ||
      element.querySelector('a[aria-label*="More"]') ||
      // Has review-like aria labels
      element.querySelector('[aria-label*="review"]')
    );

    const hasReviewContent = (
      // Text mentions reviews, ratings, or feedback
      /\b(?:review|rating|star|recommend|experience|service|staff|food|place)\b/i.test(element.textContent) &&
      // Has reasonable length for review content
      element.textContent.trim().length > 50
    );

    return hasReviewStructure || hasReviewContent;
  },

  analyzeElementForReview(element, index) {
    const text = element.textContent?.trim() || '';
    const innerHTML = element.innerHTML || '';
    
    const analysis = {
      element,
      text,
      index,
      isReview: false,
      confidence: 0,
      indicators: {
        hasRating: false,
        hasTemporal: false,
        hasReviewContent: false,
        hasReviewMeta: false,
        hasAuthor: false,
        hasBusinessRef: false
      },
      extractedData: {}
    };

    // Check aria-labels for review indicators (like "Read more of John's review")
    const allAriaLabels = element.querySelectorAll('[aria-label]');
    const ariaText = Array.from(allAriaLabels).map(el => el.getAttribute('aria-label')).join(' ');
    
    if (/review/i.test(ariaText)) {
      analysis.confidence += 15; // Bonus for aria-label containing "review"
      
      // Extract author name from aria-label like "Read more of Fatimah Mazlan's review"
      const authorMatch = ariaText.match(/(?:of|by)\s+([^']+)'s?\s+review/i);
      if (authorMatch) {
        analysis.indicators.hasAuthor = true;
        analysis.extractedData.author_name = authorMatch[1].trim();
        analysis.confidence += 20;
      }
    }

    // Check for rating indicators
    this.reviewIndicators.rating.forEach(pattern => {
      if (pattern.test(text) || pattern.test(innerHTML)) {
        analysis.indicators.hasRating = true;
        analysis.confidence += 25;
        
        // Try to extract rating value
        const ratingMatch = text.match(/\b([1-5](?:\.\d)?)/);
        if (ratingMatch) {
          analysis.extractedData.rating = parseFloat(ratingMatch[1]);
        }
      }
    });

    // Check for temporal indicators
    this.reviewIndicators.temporal.forEach(pattern => {
      if (pattern.test(text)) {
        analysis.indicators.hasTemporal = true;
        analysis.confidence += 20;
        
        // Try to extract date information
        const timeMatch = text.match(pattern);
        if (timeMatch) {
          analysis.extractedData.time = timeMatch[0];
        }
      }
    });

    // Check for review content indicators
    this.reviewIndicators.reviewContent.forEach(pattern => {
      if (pattern.test(text)) {
        analysis.indicators.hasReviewContent = true;
        analysis.confidence += 15;
      }
    });

    // Check for review meta indicators
    this.reviewIndicators.reviewMeta.forEach(pattern => {
      if (pattern.test(text)) {
        analysis.indicators.hasReviewMeta = true;
        analysis.confidence += 10;
      }
    });

    // Look for author information in nearby elements
    const authorData = this.findAuthorInformation(element);
    if (authorData) {
      analysis.indicators.hasAuthor = true;
      analysis.confidence += 20;
      analysis.extractedData.author_name = authorData;
    }

    // Look for business references
    const businessRef = this.findBusinessReference(element);
    if (businessRef) {
      analysis.indicators.hasBusinessRef = true;
      analysis.confidence += 10;
      analysis.extractedData.business_name = businessRef;
    }

    // Additional heuristics based on element structure
    analysis.confidence += this.analyzeStructuralIndicators(element);

    // Bonus for Google Maps review containers
    if (this.isGoogleMapsReviewContainer(element)) {
      analysis.confidence += 15;
      console.log('🗺️ Google Maps review container detected (+15 bonus)');
    }

    // Determine if this is likely a review
    analysis.isReview = analysis.confidence >= 40;
    
    if (analysis.isReview) {
      analysis.extractedData.text = text;
      analysis.extractedData.confidence = analysis.confidence;
      
      console.log(`🎯 Review detected (${analysis.confidence}% confidence):`, {
        text: text.substring(0, 100) + '...',
        indicators: analysis.indicators,
        extractedData: analysis.extractedData
      });
    }

    return analysis;
  },

  findAuthorInformation(element) {
    // Look for author information in nearby elements
    const searchRadius = 3; // How many parent/sibling levels to check
    const candidates = [];
    
    // Check parent and sibling elements
    let current = element.parentElement;
    for (let i = 0; i < searchRadius && current; i++) {
      // Check all children of current element for author-like text
      const children = Array.from(current.children);
      children.forEach(child => {
        const text = child.textContent?.trim() || '';
        
        // Look for name patterns
        if (this.looksLikePersonName(text)) {
          candidates.push({ text, distance: i, element: child });
        }
      });
      
      current = current.parentElement;
    }

    // Return the most likely candidate (closest with best name pattern)
    candidates.sort((a, b) => a.distance - b.distance);
    return candidates.length > 0 ? candidates[0].text : null;
  },

  looksLikePersonName(text) {
    // Basic heuristics for person names
    if (!text || text.length < 2 || text.length > 50) return false;
    
    // Check for common name patterns
    const namePatterns = [
      /^[A-Z][a-z]+ [A-Z][a-z]+$/, // "John Doe"
      /^[A-Z][a-z]+\s+[A-Z]\.$/, // "John D."
      /^[A-Z][a-z]+$/, // "John"
      /^[A-Z]{2,4}$/ // Initials like "JD" or "ABC"
    ];

    const isNamePattern = namePatterns.some(pattern => pattern.test(text));
    
    // Exclude common non-name text
    const nonNameIndicators = [
      /\b(?:review|rating|star|comment|post|article|page|website|link|button|menu|nav)\b/i,
      /\b(?:click|read|more|less|show|hide|edit|delete|share|like|follow)\b/i,
      /\d{2,}/, // Numbers longer than 2 digits
      /@|\.com|\.org|http/i, // Email/URL indicators
    ];

    const hasNonNameIndicators = nonNameIndicators.some(pattern => pattern.test(text));
    
    return isNamePattern && !hasNonNameIndicators;
  },

  findBusinessReference(element) {
    // Look for business name references in the page
    const pageTitle = document.title;
    const h1Elements = Array.from(document.querySelectorAll('h1'));
    const text = element.textContent?.trim() || '';
    
    // Strategy 1: Check if review text mentions business name from page
    let businessName = null;
    
    if (h1Elements.length > 0) {
      businessName = h1Elements[0].textContent?.trim();
    } else if (pageTitle) {
      // Clean up page title (remove common suffixes)
      businessName = pageTitle.replace(/\s*[-|·]\s*.*$/, '').trim();
    }

    // Strategy 2: Look for "Thanks [BusinessName]!" patterns in review text
    const thanksPattern = /(?:thanks?|thank you)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)[!\s]/i;
    const thanksMatch = text.match(thanksPattern);
    if (thanksMatch) {
      const mentionedBusiness = thanksMatch[1].trim();
      // If it's a reasonable business name length
      if (mentionedBusiness.length >= 3 && mentionedBusiness.length <= 50) {
        return mentionedBusiness;
      }
    }

    // Strategy 3: Look for @ mentions or direct business references
    const businessMention = text.match(/@([A-Z][a-zA-Z\s]{2,30})/);
    if (businessMention) {
      return businessMention[1].trim();
    }

    return businessName && businessName.length > 2 ? businessName : null;
  },

  analyzeStructuralIndicators(element) {
    let bonus = 0;
    
    // Check element positioning and structure
    const rect = element.getBoundingClientRect();
    const text = element.textContent?.trim() || '';
    
    // Bonus for elements with good text length for reviews
    if (text.length > 50 && text.length < 2000) {
      bonus += 5;
    }
    
    // Bonus for elements that appear to be in a list or grid structure
    const parent = element.parentElement;
    if (parent) {
      const siblings = Array.from(parent.children).filter(child => 
        child.textContent?.trim().length > 20
      );
      
      if (siblings.length >= 2 && siblings.length <= 20) {
        bonus += 10; // Likely part of a review list
      }
    }

    // Bonus for elements with reasonable positioning
    if (rect.width > 200 && rect.height > 30) {
      bonus += 5;
    }

    // Special bonus for elements with review-specific attributes
    if (element.hasAttribute('data-id') || element.hasAttribute('jsdata')) {
      bonus += 8; // Google Maps style attributes
    }

    // Bonus for elements containing expandable content
    if (element.querySelector('a[aria-label*="More"]') || 
        element.querySelector('a[aria-label*="more"]') ||
        element.querySelector('.more') ||
        element.querySelector('[role="button"]')) {
      bonus += 7; // Has expandable content (typical for long reviews)
    }

    return bonus;
  },

  isVisible(element) {
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    
    return rect.width > 0 && 
           rect.height > 0 && 
           style.display !== 'none' && 
           style.visibility !== 'hidden' && 
           parseFloat(style.opacity) > 0;
  },

  isNavigationElement(element) {
    const text = element.textContent?.trim().toLowerCase() || '';
    const navKeywords = [
      'menu', 'navigation', 'nav', 'footer', 'header', 'sidebar',
      'login', 'signup', 'sign in', 'sign up', 'register',
      'home', 'about', 'contact', 'privacy', 'terms',
      'cookie', 'search', 'filter', 'sort'
    ];
    
    return navKeywords.some(keyword => text.includes(keyword)) ||
           element.closest('nav, header, footer, aside');
  }
};

// Data extraction functions for pipeline processing
function extractReviewData() {
  console.log('🚀 Extracting review data from page using semantic detection...');
  
  try {
    const url = window.location.href;
    console.log('📍 Current URL:', url);
    
    // Use semantic detection first
    const detectedReviews = ReviewDetector.detectReviewElements();
    
    if (detectedReviews.length > 0) {
      // Return the highest confidence review
      const bestReview = detectedReviews.reduce((best, current) => 
        current.confidence > best.confidence ? current : best
      );
      
      console.log('✅ Best review found:', bestReview.extractedData);
      return bestReview.extractedData;
    }
    
    // Fallback to platform-specific extraction if semantic detection fails
    if (url.includes('google.com/maps') || url.includes('maps.google.com')) {
      console.log('🗺️ Fallback: Google Maps extraction');
      return extractGoogleMapsData();
    } else if (url.includes('yelp.com')) {
      console.log('🍽️ Fallback: Yelp extraction');
      return extractYelpData();
    } else if (url.includes('tripadvisor.com')) {
      console.log('✈️ Fallback: TripAdvisor extraction');
      return extractTripAdvisorData();
    } else {
      console.log('🔧 Fallback: Generic extraction');
      return extractGenericReviewData();
    }
    
  } catch (error) {
    console.error('❌ Data extraction failed:', error);
    return null;
  }
}

function extractGoogleMapsData() {
  console.log('🗺️ Enhanced Google Maps extraction (no class dependencies)...');
  const data = {};
  
  // Business name - use semantic search instead of classes
  const businessName = findBusinessNameGoogleMaps();
  if (businessName) data.business_name = businessName;
  
  // Address - look for address patterns
  const address = findAddressGoogleMaps();
  if (address) data.address = address;
  
  // Rating and review count - semantic detection
  const ratingInfo = findRatingInfoGoogleMaps();
  if (ratingInfo.rating) data.avg_rating = ratingInfo.rating;
  if (ratingInfo.reviewCount) data.num_of_reviews = ratingInfo.reviewCount;
  
  // Category - find business type information
  const category = findCategoryGoogleMaps();
  if (category) data.category = category;
  
  // Individual review - use semantic detection
  const reviewData = findReviewDataGoogleMaps();
  if (reviewData.author) data.author_name = reviewData.author;
  if (reviewData.text) data.text = reviewData.text;
  if (reviewData.rating) data.rating = reviewData.rating;
  if (reviewData.time) data.time = reviewData.time;
  
  console.log('📍 Google Maps extracted data:', data);
  return data;
}

function findBusinessNameGoogleMaps() {
  // Look for the main business name (usually the largest heading)
  const headings = Array.from(document.querySelectorAll('h1, h2, [role="heading"]'));
  
  for (const heading of headings) {
    const text = heading.textContent?.trim();
    if (text && text.length > 2 && text.length < 100 && 
        !text.includes('Review') && !text.includes('reviews') &&
        !text.includes('Photos') && !text.includes('About')) {
      return text;
    }
  }
  
  // Fallback: try page title
  return document.title.split(' - ')[0]?.trim();
}

function findAddressGoogleMaps() {
  // Look for address patterns in text content
  const allText = Array.from(document.querySelectorAll('*')).map(el => el.textContent?.trim()).join(' ');
  
  // Common address patterns
  const addressPatterns = [
    /\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Circle|Cir|Court|Ct)(?:\s+[A-Za-z\s]+)?(?:\s+\d{5})?/i,
    /\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}/i
  ];
  
  for (const pattern of addressPatterns) {
    const match = allText.match(pattern);
    if (match) return match[0].trim();
  }
  
  return null;
}

function findRatingInfoGoogleMaps() {
  const result = { rating: null, reviewCount: null };
  
  // Look for rating patterns (numbers near stars or "rating")
  const allElements = Array.from(document.querySelectorAll('*'));
  
  for (const element of allElements) {
    const text = element.textContent?.trim() || '';
    const ariaLabel = element.getAttribute('aria-label') || '';
    
    // Look for rating patterns
    const ratingMatch = text.match(/([1-5](?:\.\d)?)\s*(?:stars?|★|⭐|out of 5|\/5)/i) ||
                       ariaLabel.match(/([1-5](?:\.\d)?)\s*(?:stars?|out of)/i);
    
    if (ratingMatch && !result.rating) {
      result.rating = parseFloat(ratingMatch[1]);
    }
    
    // Look for review count patterns
    const reviewCountMatch = text.match(/([\d,]+)\s*reviews?/i);
    if (reviewCountMatch && !result.reviewCount) {
      result.reviewCount = parseInt(reviewCountMatch[1].replace(/,/g, ''));
    }
  }
  
  return result;
}

function findCategoryGoogleMaps() {
  // Look for business category/type information
  const categoryKeywords = ['Restaurant', 'Hotel', 'Store', 'Service', 'Bar', 'Cafe', 'Shop', 'Center', 'Park', 'Museum'];
  
  const allElements = Array.from(document.querySelectorAll('*'));
  
  for (const element of allElements) {
    const text = element.textContent?.trim() || '';
    
    for (const keyword of categoryKeywords) {
      if (text.includes(keyword) && text.length < 50) {
        return text;
      }
    }
  }
  
  return null;
}

function findReviewDataGoogleMaps() {
  const result = { author: null, text: null, rating: null, time: null };
  
  // Use the semantic detector to find reviews
  const reviews = ReviewDetector.detectReviewElements();
  
  if (reviews.length > 0) {
    const bestReview = reviews[0]; // Take the first (highest confidence) review
    
    result.text = bestReview.extractedData.text;
    result.author = bestReview.extractedData.author_name;
    result.rating = bestReview.extractedData.rating;
    result.time = bestReview.extractedData.time;
  }
  
  return result;
}

function extractYelpData() {
  console.log('🍽️ Enhanced Yelp extraction (semantic-based)...');
  const data = {};
  
  // Business name - semantic approach
  const businessName = findBusinessNameGeneric();
  if (businessName) data.business_name = businessName;
  
  // Use semantic detector for reviews
  const reviews = ReviewDetector.detectReviewElements();
  if (reviews.length > 0) {
    const bestReview = reviews[0];
    Object.assign(data, bestReview.extractedData);
  }
  
  console.log('🍽️ Yelp extracted data:', data);
  return data;
}

function extractTripAdvisorData() {
  console.log('✈️ Enhanced TripAdvisor extraction (semantic-based)...');
  const data = {};
  
  // Business name - semantic approach
  const businessName = findBusinessNameGeneric();
  if (businessName) data.business_name = businessName;
  
  // Use semantic detector for reviews
  const reviews = ReviewDetector.detectReviewElements();
  if (reviews.length > 0) {
    const bestReview = reviews[0];
    Object.assign(data, bestReview.extractedData);
  }
  
  console.log('✈️ TripAdvisor extracted data:', data);
  return data;
}

function extractGenericReviewData() {
  console.log('🔧 Enhanced Generic extraction (semantic-based)...');
  const data = {};
  
  // Business name - semantic approach
  const businessName = findBusinessNameGeneric();
  if (businessName) data.business_name = businessName;
  
  // Use semantic detector for reviews
  const reviews = ReviewDetector.detectReviewElements();
  if (reviews.length > 0) {
    const bestReview = reviews[0];
    Object.assign(data, bestReview.extractedData);
  } else {
    // Last resort fallback
    const fallbackText = findAnyMeaningfulText();
    if (fallbackText) {
      data.text = fallbackText;
      data.author_name = "Anonymous";
    }
  }
  
  console.log('🔧 Generic extracted data:', data);
  return data;
}

function findBusinessNameGeneric() {
  // Try multiple strategies to find business name
  const strategies = [
    () => document.querySelector('h1')?.textContent?.trim(),
    () => document.querySelector('[role="heading"][aria-level="1"]')?.textContent?.trim(),
    () => document.querySelector('title')?.textContent?.split('-')[0]?.trim(),
    () => document.title?.split('|')[0]?.trim(),
    () => document.title?.split('-')[0]?.trim()
  ];
  
  for (const strategy of strategies) {
    const result = strategy();
    if (result && result.length > 2 && result.length < 100) {
      return result;
    }
  }
  
  return null;
}

function findAnyMeaningfulText() {
  // Find any meaningful text content as last resort
  const elements = Array.from(document.querySelectorAll('p, div, span'));
  
  for (const element of elements) {
    const text = element.textContent?.trim();
    if (text && text.length > 50 && text.length < 1000 && 
        ReviewDetector.isVisible(element) && 
        !ReviewDetector.isNavigationElement(element)) {
      return text;
    }
  }
  
  return null;
}

// Visual review highlighting system
const ReviewHighlighter = {
  highlightedElements: new Set(),
  
  highlightDetectedReviews() {
    console.log('🎨 Highlighting detected reviews...');
    
    const reviews = ReviewDetector.detectReviewElements();
    
    reviews.forEach((review, index) => {
      if (review.confidence >= 50) { // Only highlight high-confidence reviews
        this.highlightElement(review.element, review.confidence, index);
      }
    });
    
    return reviews.length;
  },
  
  highlightElement(element, confidence, index) {
    if (this.highlightedElements.has(element)) return;
    
    this.highlightedElements.add(element);
    
    // Create highlight overlay
    const highlight = document.createElement('div');
    highlight.className = 'review-detector-highlight';
    highlight.style.cssText = `
      position: absolute;
      background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
      border: 2px solid ${confidence >= 70 ? '#22c55e' : confidence >= 60 ? '#f59e0b' : '#3b82f6'};
      border-radius: 8px;
      pointer-events: all;
      z-index: 9998;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
      cursor: pointer;
    `;
    
    // Add confidence badge
    const badge = document.createElement('div');
    badge.style.cssText = `
      position: absolute;
      top: -12px;
      right: -12px;
      background: ${confidence >= 70 ? '#22c55e' : confidence >= 60 ? '#f59e0b' : '#3b82f6'};
      color: white;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: bold;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;
    badge.textContent = `${Math.round(confidence)}%`;
    highlight.appendChild(badge);
    
    // Add validation status badge (initially hidden)
    const validationBadge = document.createElement('div');
    validationBadge.className = 'validation-badge';
    validationBadge.style.cssText = `
      position: absolute;
      top: -12px;
      left: -12px;
      background: #6b7280;
      color: white;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: bold;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      display: none;
    `;
    validationBadge.textContent = 'Checking...';
    highlight.appendChild(validationBadge);
    
    // Position the highlight
    const updatePosition = () => {
      const rect = element.getBoundingClientRect();
      highlight.style.top = `${rect.top + window.scrollY - 2}px`;
      highlight.style.left = `${rect.left + window.scrollX - 2}px`;
      highlight.style.width = `${rect.width + 4}px`;
      highlight.style.height = `${rect.height + 4}px`;
    };
    
    updatePosition();
    document.body.appendChild(highlight);
    
    // Update position on scroll
    const scrollHandler = () => updatePosition();
    window.addEventListener('scroll', scrollHandler);
    
    // Add click handler for AI validation
    highlight.addEventListener('click', async (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      // Show loading state
      validationBadge.style.display = 'block';
      validationBadge.textContent = 'Checking...';
      validationBadge.style.background = '#6b7280';
      
      try {
        await this.validateReviewWithAI(element, validationBadge);
      } catch (error) {
        console.error('Validation failed:', error);
        validationBadge.textContent = 'Error';
        validationBadge.style.background = '#ef4444';
      }
    });
    
    // Remove highlight after 15 seconds (longer to allow validation)
    setTimeout(() => {
      highlight.remove();
      window.removeEventListener('scroll', scrollHandler);
      this.highlightedElements.delete(element);
    }, 15000);
  },

  async validateReviewWithAI(element, statusBadge) {
    const text = element.textContent?.trim() || '';
    
    // Extract additional data from the element
    const reviewData = this.extractReviewDataFromElement(element);
    
    console.log('🤖 Validating review with AI:', reviewData);
    
    try {
      console.log('🚀 STARTING AI VALIDATION REQUEST');
      console.log('📤 Request payload:', {
        text: reviewData.text.substring(0, 100) + '...',
        business_name: reviewData.business_name,
        author_name: reviewData.author_name,
        rating: reviewData.rating,
        time: reviewData.time,
        additional_data: reviewData.additional_data
      });

      const response = await fetch('http://localhost:8000/review/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: reviewData.text,
          business_name: reviewData.business_name,
          author_name: reviewData.author_name,
          rating: reviewData.rating,
          time: reviewData.time,
          additional_data: reviewData.additional_data
        })
      });

      console.log('📡 Response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ API Error Response:', errorText);
        throw new Error(`API Error: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      
      console.log('🎯 AI VALIDATION COMPLETE!');
      console.log('✅ Full Result:', result);
      console.log(`📊 Prediction: ${result.is_valid ? 'VALID' : 'INVALID'} (${Math.round(result.confidence * 100)}% confidence)`);
      console.log(`⚡ Processing Time: ${result.processing_time_ms}ms`);
      console.log(`🤖 Model: ${result.model_version}`);
      
      // Update the status badge
      if (result.is_valid) {
        statusBadge.textContent = `VALID (${Math.round(result.confidence * 100)}%)`;
        statusBadge.style.background = '#22c55e';
      } else {
        statusBadge.textContent = `INVALID (${Math.round(result.confidence * 100)}%)`;
        statusBadge.style.background = '#ef4444';
      }
      
      // Send result to popup
      chrome.runtime.sendMessage({
        action: 'aiValidationResult',
        result: result,
        reviewText: text.substring(0, 100) + '...'
      });
      
    } catch (error) {
      console.error('❌ AI validation failed:', error);
      statusBadge.textContent = 'API Error';
      statusBadge.style.background = '#ef4444';
    }
  },

  extractReviewDataFromElement(element) {
    const text = element.textContent?.trim() || '';
    
    // Try to extract author name from aria-labels
    let authorName = null;
    const ariaLabels = element.querySelectorAll('[aria-label]');
    for (const ariaEl of ariaLabels) {
      const ariaText = ariaEl.getAttribute('aria-label');
      const authorMatch = ariaText?.match(/(?:of|by)\s+([^']+)'s?\s+review/i);
      if (authorMatch) {
        authorName = authorMatch[1].trim();
        break;
      }
    }
    
    // Try to extract rating
    let rating = null;
    const ratingMatch = text.match(/\b([1-5](?:\.\d)?)\s*(?:stars?|\/5|out of 5)/i) ||
                       text.match(/\b([1-9]|10)\/10\b/i);
    if (ratingMatch) {
      rating = parseFloat(ratingMatch[1]);
      if (text.includes('/10')) {
        rating = rating / 2; // Convert 10-point to 5-point scale
      }
    }
    
    // Try to extract business name
    let businessName = null;
    const thanksMatch = text.match(/(?:thanks?|thank you)\s+([A-Z][a-zA-Z\s]{2,30})[!\s]/i);
    if (thanksMatch) {
      businessName = thanksMatch[1].trim();
    } else {
      // Fallback to page title
      businessName = document.title.split(' - ')[0]?.trim();
    }
    
    return {
      text: text,
      business_name: businessName,
      author_name: authorName,
      rating: rating,
      time: null, // Could be extracted with more complex parsing
      additional_data: {
        word_count: text.split(' ').length,
        text_length: text.length,
        source_url: window.location.href
      }
    };
  },
  
  clearHighlights() {
    document.querySelectorAll('.review-detector-highlight').forEach(el => el.remove());
    this.highlightedElements.clear();
  }
};

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 Received message:', message);
  
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
  } else if (message.action === 'extractReviewData') {
    // Extract review data from current page using enhanced semantic detection
    console.log('🔍 Starting enhanced review extraction...');
    
    const extractedData = extractReviewData();
    
    if (extractedData && Object.keys(extractedData).length > 0) {
      console.log('✅ Successfully extracted review data:', extractedData);
      sendResponse({
        success: true,
        data: extractedData,
        message: `Extracted ${Object.keys(extractedData).length} data fields`
      });
    } else {
      console.log('❌ No review data found');
      sendResponse({
        success: false,
        error: 'No review data found on this page',
        message: 'Try navigating to a page with reviews (Google Maps, Yelp, etc.)'
      });
    }
  } else if (message.action === 'detectReviews') {
    // Detect and highlight all reviews on the page
    console.log('🎯 Starting review detection...');
    
    const reviews = ReviewDetector.detectReviewElements();
    const highlightCount = ReviewHighlighter.highlightDetectedReviews();
    
    sendResponse({
      success: true,
      reviewsFound: reviews.length,
      highlightedReviews: highlightCount,
      reviews: reviews.map(r => ({
        confidence: r.confidence,
        text: r.text?.substring(0, 100) + '...',
        indicators: r.indicators,
        extractedData: r.extractedData
      }))
    });
  } else if (message.action === 'clearHighlights') {
    // Clear all review highlights
    ReviewHighlighter.clearHighlights();
    sendResponse({ success: true, message: 'Highlights cleared' });
  } else if (message.action === 'analyzePageForReviews') {
    // Comprehensive page analysis
    console.log('📊 Analyzing page for review content...');
    
    const allElements = ReviewDetector.getAllTextElements();
    const reviews = ReviewDetector.detectReviewElements();
    const url = window.location.href;
    
    const analysis = {
      url: url,
      pageTitle: document.title,
      totalTextElements: allElements.length,
      reviewsDetected: reviews.length,
      highConfidenceReviews: reviews.filter(r => r.confidence >= 70).length,
      mediumConfidenceReviews: reviews.filter(r => r.confidence >= 50 && r.confidence < 70).length,
      lowConfidenceReviews: reviews.filter(r => r.confidence < 50).length,
      topReviews: reviews.slice(0, 3).map(r => ({
        confidence: r.confidence,
        text: r.text?.substring(0, 150) + '...',
        indicators: r.indicators
      }))
    };
    
    console.log('📊 Page analysis complete:', analysis);
    sendResponse({
      success: true,
      analysis: analysis
    });
  }
  
  // Return true to indicate we will send a response asynchronously
  return true;
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