// ==================== Main JavaScript File ====================

// Hamburger Menu Toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking on a nav link
        document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        }));
    }
});

// ==================== Utility Functions ====================

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <img src="https://img.icons8.com/color/24/000000/${type === 'success' ? 'checkmark' : type === 'error' ? 'error' : 'info'}.png" alt="${type}">
            <span>${message}</span>
        </div>
        <button class="notification-close">&times;</button>
    `;
    
    // Add styles for notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#48bb78' : type === 'error' ? '#f56565' : '#4299e1'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        z-index: 10000;
        max-width: 400px;
        animation: slideIn 0.3s ease-out;
    `;
    
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        margin-left: 15px;
    `;
    
    const content = notification.querySelector('.notification-content');
    content.style.cssText = `
        display: flex;
        align-items: center;
        gap: 10px;
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    const autoRemove = setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
    
    // Manual close
    closeBtn.addEventListener('click', () => {
        clearTimeout(autoRemove);
        notification.remove();
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Validate image file
function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/tif'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.tif')) {
        showNotification('Please select a valid image file (JPG, PNG, JPEG, TIF)', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File size must be less than 50MB', 'error');
        return false;
    }
    
    return true;
}

// Create loading overlay
function createLoadingOverlay(message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner">
                <img src="https://img.icons8.com/color/64/000000/brain-3.png" alt="Loading">
            </div>
            <h3>${message}</h3>
            <p>Please wait while our AI processes your image</p>
        </div>
    `;
    
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        backdrop-filter: blur(5px);
    `;
    
    const content = overlay.querySelector('.loading-content');
    content.style.cssText = `
        background: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    `;
    
    const spinner = overlay.querySelector('.loading-spinner');
    spinner.style.cssText = `
        margin-bottom: 20px;
        animation: bounce 2s infinite;
    `;
    
    document.body.appendChild(overlay);
    return overlay;
}

// Remove loading overlay
function removeLoadingOverlay(overlay) {
    if (overlay && overlay.parentNode) {
        overlay.remove();
    }
}

// ==================== Image Upload Handlers ====================

// Enhanced drag and drop functionality
function setupDragAndDrop(uploadSection, fileInput, onFileSelect) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadSection.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, unhighlight, false);
    });
    
    uploadSection.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadSection.classList.add('dragover');
    }
    
    function unhighlight(e) {
        uploadSection.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            if (validateImageFile(file)) {
                fileInput.files = files;
                onFileSelect(file);
            }
        }
    }
}

// ==================== API Helpers ====================

// Generic API call function
async function makeAPICall(url, formData, onProgress = null) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// ==================== Local Storage Helpers ====================

// Save result to local storage
function saveResult(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.warn('Could not save to localStorage:', error);
    }
}

// Get result from local storage
function getResult(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.warn('Could not read from localStorage:', error);
        return null;
    }
}

// Clear old results (keep only last 5)
function clearOldResults(prefix) {
    try {
        const keys = Object.keys(localStorage).filter(key => key.startsWith(prefix));
        if (keys.length > 5) {
            keys.sort().slice(0, keys.length - 5).forEach(key => {
                localStorage.removeItem(key);
            });
        }
    } catch (error) {
        console.warn('Could not clear old results:', error);
    }
}

// ==================== Analytics ====================

// Simple analytics tracking
function trackEvent(eventName, data = {}) {
    try {
        // You can integrate with Google Analytics, Mixpanel, etc.
        console.log('Event tracked:', eventName, data);
        
        // Example Google Analytics tracking (if gtag is available)
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, data);
        }
    } catch (error) {
        console.warn('Analytics tracking failed:', error);
    }
}

// ==================== Error Handling ====================

// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred. Please try again.', 'error');
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    showNotification('An unexpected error occurred. Please try again.', 'error');
});

// ==================== Performance Monitoring ====================

// Monitor page load performance
window.addEventListener('load', function() {
    if ('performance' in window) {
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log('Page load time:', loadTime, 'ms');
        
        trackEvent('page_load_time', {
            load_time: loadTime,
            page: window.location.pathname
        });
    }
});

// ==================== Accessibility Improvements ====================

// Add keyboard navigation support
document.addEventListener('keydown', function(e) {
    // Escape key to close modals/overlays
    if (e.key === 'Escape') {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            // Don't allow closing loading overlays with escape
            return;
        }
    }
    
    // Enter key to trigger file upload on focused upload areas
    if (e.key === 'Enter' && e.target.classList.contains('upload-section')) {
        const fileInput = e.target.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.click();
        }
    }
});

// ==================== Service Worker Registration (for PWA) ====================

if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}

// ==================== Export for use in other files ====================
window.AppUtils = {
    showNotification,
    formatFileSize,
    validateImageFile,
    createLoadingOverlay,
    removeLoadingOverlay,
    setupDragAndDrop,
    makeAPICall,
    saveResult,
    getResult,
    clearOldResults,
    trackEvent
};