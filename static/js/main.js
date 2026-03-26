/**
 * Emploify.io - Global JS Utilities
 */

// ── CSRF token injection for all fetch() calls ──
(function() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (!meta) return;
    const csrfToken = meta.getAttribute('content');
    const _origFetch = window.fetch;
    window.fetch = function(url, opts) {
        opts = opts || {};
        const method = (opts.method || 'GET').toUpperCase();
        if (['POST', 'PUT', 'DELETE', 'PATCH'].includes(method)) {
            if (opts.body instanceof FormData) {
                // FormData: append as field
                opts.body.append('csrf_token', csrfToken);
            } else {
                // JSON/other: set header
                opts.headers = opts.headers || {};
                if (typeof opts.headers.set === 'function') {
                    opts.headers.set('X-CSRFToken', csrfToken);
                } else {
                    opts.headers['X-CSRFToken'] = csrfToken;
                }
            }
        }
        return _origFetch.call(this, url, opts);
    };
})();

const Emploify = {
    /**
     * Disable a button, show spinner + loading text
     */
    lockButton(btn, text) {
        if (!btn || btn.dataset.locked === 'true') return;
        btn.dataset.locked = 'true';
        btn.dataset.originalHtml = btn.innerHTML;
        btn.disabled = true;
        btn.classList.add('opacity-75', 'cursor-not-allowed');
        btn.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i>${text || 'Processing...'}`;
    },

    /**
     * Restore original button HTML
     */
    unlockButton(btn) {
        if (!btn) return;
        btn.disabled = false;
        btn.dataset.locked = 'false';
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
        if (btn.dataset.originalHtml) {
            btn.innerHTML = btn.dataset.originalHtml;
        }
    },

    /**
     * Toast notification replacing alert() calls
     */
    showToast(msg, type = 'info') {
        const container = document.getElementById('toast-container') || (() => {
            const el = document.createElement('div');
            el.id = 'toast-container';
            el.className = 'fixed top-4 right-4 z-[9999] space-y-2 max-w-sm';
            document.body.appendChild(el);
            return el;
        })();

        const colors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };

        const toast = document.createElement('div');
        toast.className = `${colors[type] || colors.info} text-white px-4 py-3 rounded-lg shadow-lg flex items-center space-x-3 transform translate-x-full transition-transform duration-300`;
        toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span class="text-sm font-medium">${msg}</span>
            <button class="ml-auto text-white opacity-70 hover:opacity-100" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>`;

        container.appendChild(toast);
        requestAnimationFrame(() => toast.classList.remove('translate-x-full'));

        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    },

    /**
     * Toggle dark/light theme
     */
    toggleTheme() {
        const html = document.documentElement;
        html.classList.toggle('dark');
        const isDark = html.classList.contains('dark');
        localStorage.setItem('emploify-theme', isDark ? 'dark' : 'light');
        this._syncThemeIcons(isDark);
    },

    _syncThemeIcons(isDark) {
        const cls = isDark ? 'fas fa-sun text-yellow-400' : 'fas fa-moon text-gray-600';
        ['theme-icon', 'theme-icon-mobile'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.className = cls;
        });
    },

    /**
     * Read theme from localStorage on page load
     */
    initTheme() {
        const saved = localStorage.getItem('emploify-theme');
        if (saved === 'dark') {
            document.documentElement.classList.add('dark');
        }
        const isDark = document.documentElement.classList.contains('dark');
        this._syncThemeIcons(isDark);
    },

    /**
     * Creates a progress simulator for progress bars
     * Returns {complete(), fail()} controls
     */
    createProgressSimulator(barEl, pctEl) {
        let progress = 0;
        let interval = null;

        interval = setInterval(() => {
            progress += Math.random() * 8 + 2;
            if (progress > 90) progress = 90;
            if (barEl) barEl.style.width = `${Math.round(progress)}%`;
            if (pctEl) pctEl.textContent = `${Math.round(progress)}%`;
        }, 300);

        return {
            complete() {
                clearInterval(interval);
                if (barEl) barEl.style.width = '100%';
                if (pctEl) pctEl.textContent = '100%';
            },
            fail() {
                clearInterval(interval);
            }
        };
    },

    /**
     * Auto-lock all form submit buttons on submit (prevents double-click)
     */
    initFormProtection() {
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', function () {
                const btn = form.querySelector('button[type="submit"]');
                if (btn && btn.dataset.locked !== 'true') {
                    Emploify.lockButton(btn, btn.textContent.trim());
                    // Unlock after 10s as safety net
                    setTimeout(() => Emploify.unlockButton(btn), 10000);
                }
            });
        });
    },

    /**
     * Keyboard shortcuts: Esc to close modals, Ctrl+K to focus search
     */
    initKeyboardShortcuts() {
        document.addEventListener('keydown', function (e) {
            // Esc to close modals
            if (e.key === 'Escape') {
                document.querySelectorAll('.fixed.inset-0:not(.hidden)').forEach(modal => {
                    modal.classList.add('hidden');
                });
            }
            // Ctrl+K to focus search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const search = document.getElementById('search') || document.querySelector('input[type="search"]');
                if (search) search.focus();
            }
        });
    }
};

// ── Resume upload (kept from original) ──
document.addEventListener('DOMContentLoaded', function () {
    Emploify.initTheme();
    Emploify.initFormProtection();
    Emploify.initKeyboardShortcuts();

    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('resume-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadContainer = document.getElementById('upload-container');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressPercent = document.getElementById('progress-percent');
    const errorMessage = document.getElementById('error-message');

    if (!uploadBtn || !fileInput) return; // Not on upload page

    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function () {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            fileNameDisplay.classList.remove('hidden');
            uploadFile(file);
        }
    });

    if (uploadContainer) {
        uploadContainer.addEventListener('dragover', function (e) {
            e.preventDefault();
            uploadContainer.classList.add('border-primary-500', 'bg-primary-50');
        });

        uploadContainer.addEventListener('dragleave', function () {
            uploadContainer.classList.remove('border-primary-500', 'bg-primary-50');
        });

        uploadContainer.addEventListener('drop', function (e) {
            e.preventDefault();
            uploadContainer.classList.remove('border-primary-500', 'bg-primary-50');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                const file = e.dataTransfer.files[0];
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                fileNameDisplay.classList.remove('hidden');
                uploadFile(file);
            }
        });
    }

    function uploadFile(file) {
        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(pdf|docx|txt)$/i)) {
            showError('Please upload a PDF, DOCX, or TXT file.');
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            showError('File size should be less than 5MB.');
            return;
        }

        progressContainer.classList.remove('hidden');
        Emploify.lockButton(uploadBtn, 'Uploading...');

        const sim = Emploify.createProgressSimulator(progressBar, progressPercent);

        const reader = new FileReader();
        reader.onload = function() {
            const base64 = reader.result.split(',')[1];
            const payload = JSON.stringify({ filename: file.name, data: base64 });
            _doUpload(payload, sim);
        };
        reader.onerror = function() {
            sim.fail();
            showError('Failed to read the file.');
            Emploify.unlockButton(uploadBtn);
        };
        reader.readAsDataURL(file);
    }

    function _doUpload(payload, sim) {
        fetch('/upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
        })
            .then(response => {
                sim.complete();
                if (!response.ok) return response.json().then(err => { throw err; });
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    Emploify.showToast('Resume parsed successfully!', 'success');
                    setTimeout(() => {
                        window.location.href = `/dashboard`;
                    }, 1000);
                } else {
                    Emploify.showToast(data.error || 'Failed to parse resume', 'error');
                }
            })
            .catch(error => {
                sim.fail();
                Emploify.showToast(error.error || 'An error occurred while processing your file', 'error');
            })
            .finally(() => {
                Emploify.unlockButton(uploadBtn);
            });
    }

    function showError(message) {
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.classList.remove('hidden');
            setTimeout(() => errorMessage.classList.add('hidden'), 5000);
        }
        Emploify.showToast(message, 'error');
    }
});
