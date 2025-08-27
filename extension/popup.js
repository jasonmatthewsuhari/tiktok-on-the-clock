document.addEventListener('DOMContentLoaded', () => {
  const toggleButton = document.getElementById('toggleButton');
  const toggleStatus = document.getElementById('toggleStatus');

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

  function updateToggleStatus(isEnabled) {
    toggleStatus.textContent = isEnabled ? 'Active' : 'Inactive';
    toggleStatus.className = 'power-state ' + (isEnabled ? 'active' : 'inactive');
    toggleButton.classList.toggle('active', isEnabled);
  }
});