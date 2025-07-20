// SafeWatch Construction Safety Monitoring Application

// Application data from the provided JSON
const appData = {
  sampleUploadHistory: [
    {
      id: 1,
      filename: "construction_site_1.jpg",
      uploadDate: "2025-01-15",
      status: "completed",
      safetyScore: 85,
      violations: ["Missing helmet on worker 2"],
      type: "image"
    },
    {
      id: 2,
      filename: "site_video_morning.mp4",
      uploadDate: "2025-01-14",
      status: "processing",
      safetyScore: null,
      violations: [],
      type: "video"
    }
  ],
  userProfile: {
    name: "John Construction Manager",
    email: "john@constructionco.com",
    company: "SafeBuild Construction",
    role: "Site Manager"
  },
  features: [
    {
      title: "Real-time Safety Monitoring",
      description: "Upload images and videos to instantly analyze PPE compliance",
      icon: "shield"
    },
    {
      title: "AI-Powered Detection",
      description: "Advanced YOLO + PPO algorithms detect safety violations",
      icon: "eye"
    },
    {
      title: "Automated Alerts",
      description: "Get instant notifications when safety issues are detected",
      icon: "bell"
    },
    {
      title: "Compliance Reports",
      description: "Generate detailed safety compliance reports for audits",
      icon: "document"
    }
  ]
};

// Application state
let currentUser = null;
let uploadHistory = [...appData.sampleUploadHistory];
let currentPage = 'home';

// Utility functions
function showPage(pageId) {
  console.log('Navigating to page:', pageId);
  
  // Hide all pages
  document.querySelectorAll('.page').forEach(page => {
    page.style.display = 'none';
  });
  
  // Show selected page
  const targetPage = document.getElementById(pageId + '-page');
  if (targetPage) {
    targetPage.style.display = 'block';
    currentPage = pageId;
    
    // Update URL hash
    window.location.hash = pageId;
    
    console.log('Successfully navigated to:', pageId);
  } else {
    console.error('Page not found:', pageId + '-page');
  }
}

function showNotification(message, type = 'success') {
  console.log('Showing notification:', message, type);
  
  // Remove existing notifications first
  document.querySelectorAll('.notification').forEach(n => n.remove());
  
  // Create notification element
  const notification = document.createElement('div');
  notification.className = `notification notification--${type}`;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 16px 24px;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-base);
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    max-width: 400px;
    animation: slideIn 0.3s ease;
    font-size: var(--font-size-sm);
  `;
  
  if (type === 'success') {
    notification.style.borderColor = 'var(--color-success)';
    notification.style.color = 'var(--color-success)';
  } else if (type === 'error') {
    notification.style.borderColor = 'var(--color-error)';
    notification.style.color = 'var(--color-error)';
  } else if (type === 'info') {
    notification.style.borderColor = 'var(--color-primary)';
    notification.style.color = 'var(--color-primary)';
  }
  
  notification.textContent = message;
  document.body.appendChild(notification);
  
  // Remove notification after 4 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.remove();
    }
  }, 4000);
}

function updateAuthUI() {
  const loginBtn = document.getElementById('login-btn');
  const logoutBtn = document.getElementById('logout-btn');
  const dashboardLink = document.querySelector('.dashboard-link');
  const userName = document.getElementById('user-name');
  const userCompany = document.getElementById('user-company');
  
  if (currentUser) {
    if (loginBtn) loginBtn.style.display = 'none';
    if (logoutBtn) logoutBtn.style.display = 'inline-flex';
    if (dashboardLink) dashboardLink.style.display = 'block';
    
    if (userName) userName.textContent = currentUser.name;
    if (userCompany) userCompany.textContent = currentUser.company;
  } else {
    if (loginBtn) loginBtn.style.display = 'inline-flex';
    if (logoutBtn) logoutBtn.style.display = 'none';
    if (dashboardLink) dashboardLink.style.display = 'none';
  }
}

function login(userData = appData.userProfile) {
  console.log('Logging in user:', userData);
  currentUser = userData;
  updateAuthUI();
  renderUploadHistory();
  showPage('dashboard');
  showNotification('Successfully logged in!');
}

function logout() {
  console.log('Logging out user');
  currentUser = null;
  updateAuthUI();
  showPage('home');
  showNotification('Successfully logged out!');
}

// Feature cards rendering
function renderFeatures() {
  const featuresGrid = document.getElementById('features-grid');
  if (!featuresGrid) return;
  
  const iconSvgs = {
    shield: '<path d="M9 12L11 14L15 10M21 12C21 16.418 16.418 21 12 21S3 16.418 3 12S7.582 3 12 3S21 7.582 21 12Z" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    eye: '<path d="M1 12S5 4 12 4S23 12 23 12S19 20 12 20S1 12 1 12Z" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2" fill="none"/>',
    bell: '<path d="M18 8A6 6 0 0 0 6 8C6 15 3 17 3 17H21S18 15 18 8Z" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/><path d="M13.73 21A2 2 0 0 1 10.27 21" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    document: '<path d="M14 2H6A2 2 0 0 0 4 4V20A2 2 0 0 0 6 22H18A2 2 0 0 0 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/><polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
  };
  
  featuresGrid.innerHTML = appData.features.map(feature => `
    <div class="feature-card">
      <div class="feature-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          ${iconSvgs[feature.icon] || iconSvgs.document}
        </svg>
      </div>
      <h3>${feature.title}</h3>
      <p>${feature.description}</p>
    </div>
  `).join('');
}

// Upload history rendering
function renderUploadHistory() {
  const uploadHistoryEl = document.getElementById('upload-history');
  if (!uploadHistoryEl) return;
  
  uploadHistoryEl.innerHTML = uploadHistory.map(item => {
    const isVideo = item.type === 'video';
    const statusClass = item.status === 'completed' ? 'status--completed' : 'status--processing';
    
    return `
      <div class="history-item">
        <div class="file-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            ${isVideo ? 
              '<polygon points="23 7 16 12 23 17 23 7" fill="currentColor"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2" stroke="currentColor" stroke-width="2" fill="none"/>' :
              '<rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke="currentColor" stroke-width="2" fill="none"/><circle cx="8.5" cy="8.5" r="1.5" stroke="currentColor" stroke-width="2" fill="none"/><polyline points="21,15 16,10 5,21" stroke="currentColor" stroke-width="2" fill="none"/>'
            }
          </svg>
        </div>
        <div class="file-info">
          <h4>${item.filename}</h4>
          <div class="file-meta">
            <span>Uploaded: ${new Date(item.uploadDate).toLocaleDateString()}</span>
            <span class="status ${statusClass}">${item.status}</span>
          </div>
          ${item.violations.length > 0 ? `
            <div class="violations">
              ${item.violations.map(violation => `<span class="violation-tag">${violation}</span>`).join('')}
            </div>
          ` : ''}
        </div>
        <div class="safety-score">
          ${item.safetyScore !== null ? `
            <div class="score">${item.safetyScore}</div>
            <div class="label">Safety Score</div>
          ` : `
            <div class="spinner"></div>
            <div class="label">Processing</div>
          `}
        </div>
      </div>
    `;
  }).join('');
}

// File upload handling
function handleFileUpload(files) {
  if (!currentUser) {
    showNotification('Please log in to upload files', 'error');
    return;
  }
  
  const validTypes = {
    image: ['image/jpeg', 'image/png', 'image/gif'],
    video: ['video/mp4', 'video/avi', 'video/quicktime']
  };
  
  const maxSize = 50 * 1024 * 1024; // 50MB
  
  Array.from(files).forEach(file => {
    // Validate file type
    const isValidImage = validTypes.image.includes(file.type);
    const isValidVideo = validTypes.video.includes(file.type);
    
    if (!isValidImage && !isValidVideo) {
      showNotification(`Invalid file type: ${file.name}`, 'error');
      return;
    }
    
    // Validate file size
    if (file.size > maxSize) {
      showNotification(`File too large: ${file.name}`, 'error');
      return;
    }
    
    // Simulate upload process
    simulateUpload(file, isValidVideo ? 'video' : 'image');
  });
}

function simulateUpload(file, type) {
  const uploadProgress = document.getElementById('upload-progress');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  
  // Show upload progress
  if (uploadProgress) uploadProgress.style.display = 'block';
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) progressText.textContent = 'Uploading...';
  
  // Simulate upload progress
  let progress = 0;
  const uploadInterval = setInterval(() => {
    progress += Math.random() * 20;
    if (progress >= 100) {
      progress = 100;
      clearInterval(uploadInterval);
      
      // Hide progress and show success
      setTimeout(() => {
        if (uploadProgress) uploadProgress.style.display = 'none';
        
        // Add to upload history
        const newUpload = {
          id: Date.now(),
          filename: file.name,
          uploadDate: new Date().toISOString().split('T')[0],
          status: 'processing',
          safetyScore: null,
          violations: [],
          type: type
        };
        
        uploadHistory.unshift(newUpload);
        renderUploadHistory();
        
        showNotification('File uploaded successfully! Processing started.');
        
        // Simulate Kaggle notebook execution
        simulateProcessing(newUpload.id);
        
      }, 500);
    }
    
    if (progressFill) progressFill.style.width = progress + '%';
    if (progressText) progressText.textContent = `Uploading... ${Math.round(progress)}%`;
  }, 200);
}

function simulateProcessing(uploadId) {
  // Simulate processing time (3-8 seconds)
  const processingTime = 3000 + Math.random() * 5000;
  
  setTimeout(() => {
    // Find and update the upload
    const uploadIndex = uploadHistory.findIndex(item => item.id === uploadId);
    if (uploadIndex !== -1) {
      uploadHistory[uploadIndex].status = 'completed';
      uploadHistory[uploadIndex].safetyScore = Math.floor(Math.random() * 30) + 70; // 70-100
      
      // Randomly add violations
      const possibleViolations = [
        'Missing helmet on worker 1',
        'No safety vest detected',
        'Improper footwear',
        'Missing eye protection',
        'Unsafe ladder usage'
      ];
      
      if (Math.random() < 0.3) { // 30% chance of violations
        uploadHistory[uploadIndex].violations = [
          possibleViolations[Math.floor(Math.random() * possibleViolations.length)]
        ];
      }
      
      renderUploadHistory();
      showNotification('Processing completed! Results are ready.');
    }
  }, processingTime);
}

// DOM Content Loaded Event
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM Content Loaded - Initializing application');
  
  // Initialize features
  renderFeatures();
  
  // Set up all event listeners
  setupEventListeners();
  
  // Handle initial page load
  const hash = window.location.hash.substring(1);
  if (hash && ['home', 'contact', 'login', 'dashboard'].includes(hash)) {
    if (hash === 'dashboard' && !currentUser) {
      showPage('login');
    } else {
      showPage(hash);
    }
  } else {
    showPage('home');
  }
  
  // Update auth UI
  updateAuthUI();
});

function setupEventListeners() {
  // Login button in navbar
  const loginBtn = document.getElementById('login-btn');
  if (loginBtn) {
    loginBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Login button clicked');
      showPage('login');
    });
  }
  
  // Logout button
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Logout button clicked');
      logout();
    });
  }
  
  // Get Started button
  const getStartedBtn = document.getElementById('get-started-btn');
  if (getStartedBtn) {
    getStartedBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Get Started button clicked');
      showPage('login');
    });
  }
  
  // Google signin button
  const googleSigninBtn = document.getElementById('google-signin');
  if (googleSigninBtn) {
    googleSigninBtn.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Google signin button clicked');
      showNotification('Connecting to Google...', 'info');
      setTimeout(() => {
        login();
      }, 1500);
    });
  }
  
  // Navigation links
  document.addEventListener('click', function(e) {
    const target = e.target.closest('a[href^="#"]');
    if (target) {
      e.preventDefault();
      const page = target.getAttribute('href').substring(1);
      console.log('Navigation link clicked:', page);
      
      // Check if dashboard access requires authentication
      if (page === 'dashboard' && !currentUser) {
        showPage('login');
        showNotification('Please log in to access the dashboard', 'error');
        return;
      }
      
      showPage(page);
    }
  });
  
  // Login form
  const loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', function(e) {
      e.preventDefault();
      console.log('Login form submitted');
      
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      
      if (!email || !password) {
        showNotification('Please fill in all fields', 'error');
        return;
      }
      
      // Simulate login process
      showNotification('Signing in...', 'info');
      setTimeout(() => {
        login();
      }, 1000);
    });
  }
  
  // Contact form
  const contactForm = document.getElementById('contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', function(e) {
      e.preventDefault();
      console.log('Contact form submitted');
      
      const name = document.getElementById('contact-name').value;
      const email = document.getElementById('contact-email').value;
      const message = document.getElementById('contact-message').value;
      
      if (!name || !email || !message) {
        showNotification('Please fill in all required fields', 'error');
        return;
      }
      
      // Simulate form submission
      showNotification('Sending message...', 'info');
      setTimeout(() => {
        contactForm.reset();
        showNotification('Message sent successfully! We\'ll get back to you soon.');
      }, 1500);
    });
  }
  
  // File upload handling
  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  
  if (uploadArea) {
    uploadArea.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Upload area clicked');
      if (!currentUser) {
        showNotification('Please log in to upload files', 'error');
        return;
      }
      if (fileInput) {
        fileInput.click();
      }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      handleFileUpload(e.dataTransfer.files);
    });
  }
  
  if (fileInput) {
    fileInput.addEventListener('change', function(e) {
      handleFileUpload(e.target.files);
    });
  }
  
  // Mobile menu toggle
  const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
  const navbarMenu = document.getElementById('navbar-menu');
  
  if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', function(e) {
      e.preventDefault();
      console.log('Mobile menu toggle clicked');
      if (navbarMenu) {
        navbarMenu.classList.toggle('mobile-menu-open');
      }
    });
  }
}

// Handle browser back/forward navigation
window.addEventListener('hashchange', function() {
  const hash = window.location.hash.substring(1);
  console.log('Hash changed to:', hash);
  if (hash && ['home', 'contact', 'login', 'dashboard'].includes(hash)) {
    if (hash === 'dashboard' && !currentUser) {
      showPage('login');
      showNotification('Please log in to access the dashboard', 'error');
    } else {
      showPage(hash);
    }
  }
});

// Add CSS animation for notifications and mobile menu
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  .mobile-menu-open {
    display: flex !important;
    flex-direction: column;
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-top: none;
    padding: 16px;
    gap: 16px;
  }
  
  @media (min-width: 769px) {
    .mobile-menu-open {
      display: flex !important;
      position: static;
      flex-direction: row;
      background: none;
      border: none;
      padding: 0;
      gap: 32px;
    }
  }
`;
document.head.appendChild(style);