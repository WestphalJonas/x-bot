(function () {
  function safeJsonParse(text, fallback) {
    try {
      return JSON.parse(text);
    } catch {
      return fallback;
    }
  }

  function getThemeMeta() {
    return document.getElementById('theme-color-meta');
  }

  function getPreferredTheme() {
    let storedTheme = null;
    try {
      storedTheme = localStorage.getItem('theme');
    } catch {
      storedTheme = null;
    }
    if (storedTheme) {
      return storedTheme;
    }
    try {
      if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        return 'light';
      }
    } catch {
      return 'dark';
    }
    return 'dark';
  }

  function applyTheme(theme, persist) {
    document.documentElement.setAttribute('data-theme', theme);
    const meta = getThemeMeta();
    if (meta) {
      meta.content = theme === 'dark' ? '#000000' : '#ffffff';
    }
    if (persist) {
      try {
        localStorage.setItem('theme', theme);
      } catch {
        // Ignore storage failures (private mode / browser policies)
      }
    }
    window.dispatchEvent(new CustomEvent('xbot:theme-changed', { detail: { theme } }));
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark', true);
  }

  function formatLocalTime(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      return date.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    } catch {
      return isoString;
    }
  }

  function formatLocalTimeShort(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      return date.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    } catch {
      return isoString;
    }
  }

  function formatLocalTimeTimeOnly(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    } catch {
      return isoString;
    }
  }

  function convertTimestampsToLocal(root) {
    const scope = root || document;
    scope.querySelectorAll('[data-timestamp]').forEach((el) => {
      const isoString = el.getAttribute('data-timestamp');
      if (!isoString) return;
      const format = el.getAttribute('data-time-format') || 'default';
      let formatted = isoString;
      if (format === 'short') {
        formatted = formatLocalTimeShort(isoString);
      } else if (format === 'time-only') {
        formatted = formatLocalTimeTimeOnly(isoString);
      } else {
        formatted = formatLocalTime(isoString);
      }
      el.textContent = formatted;
    });
  }

  function initAnalyticsCharts(root) {
    if (typeof Chart === 'undefined') {
      return;
    }

    const scope = root || document;
    const chartDataNodes = scope.querySelectorAll('#hourlyChartData');

    chartDataNodes.forEach((dataEl) => {
      const section = dataEl.closest('.analytics-section') || dataEl.parentElement;
      if (!section) return;
      const canvas = section.querySelector('#hourlyTokenChart');
      if (!canvas) return;
      const rawData = safeJsonParse(dataEl.textContent, []);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      if (canvas._xbotChartInstance) {
        canvas._xbotChartInstance.destroy();
      }

      const style = getComputedStyle(document.documentElement);
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const accentBlue = style.getPropertyValue('--accent-blue').trim() || '#1d9bf0';
      const textSecondary = style.getPropertyValue('--text-secondary').trim() || '#71767b';
      const borderColor = style.getPropertyValue('--border-color').trim() || '#2f3336';

      const labels = rawData.map((d) => {
        const hour = (d.hour || '').split(' ')[1] || '';
        return hour.substring(0, 5);
      });
      const values = rawData.map((d) => d.tokens || 0);

      canvas._xbotChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Tokens',
              data: values,
              backgroundColor: accentBlue,
              borderColor: accentBlue,
              borderWidth: 0,
              borderRadius: 4,
              borderSkipped: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: isDark ? '#1d1f23' : '#ffffff',
              titleColor: isDark ? '#e7e9ea' : '#0f1419',
              bodyColor: isDark ? '#e7e9ea' : '#0f1419',
              borderColor,
              borderWidth: 1,
              cornerRadius: 8,
              padding: 12,
              displayColors: false,
              callbacks: {
                label(context) {
                  return `${context.parsed.y.toLocaleString()} tokens`;
                },
              },
            },
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: {
                color: textSecondary,
                font: {
                  size: 11,
                  family: "'SF Mono', 'Cascadia Code', 'Fira Code', monospace",
                },
                maxRotation: 0,
              },
              border: { display: false },
            },
            y: {
              beginAtZero: true,
              grid: {
                color: borderColor,
                drawTicks: false,
              },
              ticks: {
                color: textSecondary,
                font: { size: 11 },
                padding: 8,
                callback(value) {
                  if (value >= 1000) {
                    return `${(value / 1000).toFixed(1)}k`;
                  }
                  return value;
                },
              },
              border: { display: false },
            },
          },
          animation: {
            duration: 400,
            easing: 'easeOutQuart',
          },
        },
      });
    });
  }

  const XBotUI = {
    applyTheme,
    toggleTheme,
    convertTimestampsToLocal,
    initAnalyticsCharts,
    formatLocalTime,
    formatLocalTimeShort,
    formatLocalTimeTimeOnly,
  };
  window.XBotUI = XBotUI;
  window.toggleTheme = toggleTheme;

  document.addEventListener('DOMContentLoaded', () => {
    convertTimestampsToLocal(document);
    initAnalyticsCharts(document);
  });

  if (document.body) {
    document.body.addEventListener('htmx:afterSwap', (event) => {
      const root = event && event.target ? event.target : document;
      convertTimestampsToLocal(root);
      initAnalyticsCharts(root);
    });
  } else {
    document.addEventListener('htmx:afterSwap', (event) => {
      const root = event && event.target ? event.target : document;
      convertTimestampsToLocal(root);
      initAnalyticsCharts(root);
    });
  }

  window.addEventListener('xbot:theme-changed', () => {
    initAnalyticsCharts(document);
  });

  let alpineComponentsRegistered = false;

  function registerAlpineComponents(Alpine) {
    if (!Alpine || alpineComponentsRegistered) {
      return;
    }
    alpineComponentsRegistered = true;

    Alpine.data('appShell', () => ({
      reloadBusy: false,
      botIsActive: null,
      botHealthBusy: false,
      botHealthReason: '',
      _botHealthInterval: null,
      _systemThemeHandler: null,
      _mediaQueryList: null,
      init() {
        try {
          applyTheme(getPreferredTheme(), false);
        } catch (error) {
          console.warn('Theme init failed', error);
        }

        if (!window.matchMedia) {
          return;
        }

        try {
          this._mediaQueryList = window.matchMedia('(prefers-color-scheme: dark)');
          this._systemThemeHandler = (e) => {
            let hasStoredTheme = false;
            try {
              hasStoredTheme = Boolean(localStorage.getItem('theme'));
            } catch {
              hasStoredTheme = false;
            }
            if (!hasStoredTheme) {
              applyTheme(e.matches ? 'dark' : 'light', false);
            }
          };

          if (typeof this._mediaQueryList.addEventListener === 'function') {
            this._mediaQueryList.addEventListener('change', this._systemThemeHandler);
          } else if (typeof this._mediaQueryList.addListener === 'function') {
            this._mediaQueryList.addListener(this._systemThemeHandler);
          }
        } catch (error) {
          console.warn('System theme listener init failed', error);
        }

        this.refreshBotHealth();
        this._botHealthInterval = window.setInterval(() => {
          this.refreshBotHealth();
        }, 10000);
      },
      toggleTheme() {
        toggleTheme();
      },
      async refreshBotHealth() {
        if (this.botHealthBusy) return;
        this.botHealthBusy = true;

        try {
          const response = await fetch('/api/bot/health', {
            method: 'GET',
            headers: { Accept: 'application/json' },
            cache: 'no-store',
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Health check failed');
          }
          this.botIsActive = data.active === true;
          this.botHealthReason = data.reason || '';
        } catch (error) {
          this.botIsActive = false;
          this.botHealthReason = error instanceof Error ? error.message : String(error);
          console.warn('Bot health check failed', error);
        } finally {
          this.botHealthBusy = false;
        }
      },
      botStatusText() {
        if (this.botIsActive === null) return 'Checking...';
        return this.botIsActive ? 'Active' : 'Inactive';
      },
      async reloadConfig() {
        const btn = this.$refs.reloadConfigBtn || document.getElementById('reload-config-btn');
        if (!btn || this.reloadBusy) return;

        const originalContent = btn.innerHTML;
        this.reloadBusy = true;
        btn.disabled = true;
        btn.innerHTML = '<span class="reload-spinner"></span><span>Reloading...</span>';

        try {
          const response = await fetch('/api/config/reload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
          });
          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.detail || 'Reload failed');
          }
          btn.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg><span>Reloaded!</span>';
          btn.classList.add('reload-success');
          setTimeout(() => {
            btn.innerHTML = originalContent;
            btn.classList.remove('reload-success');
          }, 2000);
        } catch (error) {
          console.error('Config reload failed:', error);
          btn.innerHTML = '<span>Error!</span>';
          btn.classList.add('reload-error');
          setTimeout(() => {
            btn.innerHTML = originalContent;
            btn.classList.remove('reload-error');
          }, 2000);
        } finally {
          this.reloadBusy = false;
          btn.disabled = false;
        }
      },
    }));

    Alpine.data('postsPage', (initialTab) => ({
      activeTab: initialTab || 'read',
      setActiveTab(tab) {
        this.activeTab = tab;
      },
    }));

    Alpine.data('dashboardPage', (initialState) => ({
      state: initialState || {},
      uptimeText: '',
      overview: null,
      overviewLoading: false,
      overviewError: '',
      actionBusy: {},
      _uptimeInterval: null,
      _overviewInterval: null,
      init() {
        this.updateUptime();
        this._uptimeInterval = window.setInterval(() => this.updateUptime(), 60000);
        this.refreshOverview();
        this._overviewInterval = window.setInterval(() => this.refreshOverview(), 10000);
        window.addEventListener('xbot:dashboard-toggle', () => {
          this.toggleSchedulerPause();
        });
        const btn = document.getElementById('pause-resume-btn');
        if (btn) {
          btn.dataset.paused = this.state.paused ? 'true' : 'false';
        }
      },
      async refreshOverview() {
        if (this.overviewLoading) return;
        this.overviewLoading = true;
        this.overviewError = '';
        try {
          const response = await fetch('/api/dashboard/overview', {
            headers: { Accept: 'application/json' },
            cache: 'no-store',
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Failed to load overview');
          }
          this.overview = data;
          const health = (data && data.health) || {};
          if (Object.prototype.hasOwnProperty.call(health, 'bot_started_at')) {
            this.state.startedAt = health.bot_started_at;
          }
          if (Object.prototype.hasOwnProperty.call(health, 'bot_stopped_at')) {
            this.state.stoppedAt = health.bot_stopped_at;
          }
          if (Object.prototype.hasOwnProperty.call(health, 'state_running')) {
            this.state.running = health.state_running;
          }
          this.updateUptime();
          this.$nextTick(() => this.scrollRecentLogsToBottom());
        } catch (error) {
          this.overviewError = error instanceof Error ? error.message : String(error);
          console.warn('Dashboard overview refresh failed', error);
        } finally {
          this.overviewLoading = false;
        }
      },
      scrollRecentLogsToBottom() {
        const el = this.$refs && this.$refs.recentLogsViewport;
        if (!el) return;
        el.scrollTop = el.scrollHeight;
      },
      updateUptime() {
        if (!this.state.startedAt) {
          this.uptimeText = '';
          return;
        }
        const started = new Date(this.state.startedAt);
        const stopPoint =
          this.state.running === false && this.state.stoppedAt
            ? new Date(this.state.stoppedAt)
            : new Date();
        if (Number.isNaN(started.getTime()) || Number.isNaN(stopPoint.getTime())) {
          this.uptimeText = '';
          return;
        }
        const diff = stopPoint - started;
        if (diff < 0) {
          this.uptimeText = '';
          return;
        }
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        let text = '';
        if (days > 0) text += `${days}d `;
        if (hours > 0 || days > 0) text += `${hours}h `;
        text += `${minutes}m`;
        this.uptimeText = this.state.running === false ? `${text} ran for` : `${text} uptime`;
      },
      async toggleSchedulerPause() {
        const btn = document.getElementById('pause-resume-btn');
        if (!btn || btn.disabled) return;

        const wasPaused = !!this.state.paused;
        const url = wasPaused ? '/api/scheduler/resume' : '/api/scheduler/pause';
        const originalContent = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = wasPaused ? 'Resuming...' : 'Pausing...';

        try {
          const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
          });
          const data = await response.json();
          if (!response.ok || data.status === 'error') {
            throw new Error(data.reason || 'Request failed');
          }
          const isPausedNow = data.status === 'paused';
          this.state.paused = isPausedNow;
          btn.dataset.paused = isPausedNow ? 'true' : 'false';
          btn.textContent = isPausedNow ? 'Resume' : 'Pause';
          if (isPausedNow) {
            const count = data.next_runs ? Object.keys(data.next_runs).length : 0;
            this.state.nextRunsCount = count;
          }
          this.refreshOverview();
        } catch (error) {
          console.error('Scheduler toggle failed:', error);
          btn.textContent = 'Error';
          setTimeout(() => {
            btn.innerHTML = originalContent;
          }, 2000);
        } finally {
          btn.disabled = false;
        }
      },
      schedulerStatusLabel() {
        return this.state.paused ? 'Paused' : 'Running';
      },
      schedulerSubtext() {
        if (this.state.paused) {
          return `Next runs captured: ${this.state.nextRunsCount || 0}`;
        }
        const nextJob = this.sortedJobs()[0];
        if (nextJob && nextJob.next_run_in_seconds != null) {
          return `Next ${this.prettyJobName(nextJob.job_id)} in ${this.formatDuration(nextJob.next_run_in_seconds)}`;
        }
        return 'Live schedule loaded';
      },
      jobs() {
        return (this.overview && this.overview.scheduler && this.overview.scheduler.jobs) || [];
      },
      sortedJobs() {
        return this.jobs()
          .slice()
          .sort((a, b) => {
            const av = a.next_run_in_seconds ?? Number.POSITIVE_INFINITY;
            const bv = b.next_run_in_seconds ?? Number.POSITIVE_INFINITY;
            return av - bv;
          });
      },
      queuePendingJobs() {
        return (this.overview && this.overview.queues && this.overview.queues.pending_jobs) || [];
      },
      queuePreview(kind) {
        if (!this.overview || !this.overview.queues) return [];
        return this.overview.queues[kind] || [];
      },
      timelineItems() {
        return (this.overview && this.overview.timeline) || [];
      },
      logLines() {
        return (this.overview && this.overview.logs && this.overview.logs.tail) || [];
      },
      todayMetrics() {
        return (this.overview && this.overview.today) || {};
      },
      pipelineMetrics() {
        return (this.overview && this.overview.pipeline) || {};
      },
      healthSnapshot() {
        return (this.overview && this.overview.health) || {};
      },
      formatDuration(seconds) {
        if (seconds == null || Number.isNaN(Number(seconds))) return 'n/a';
        const total = Math.max(0, Math.floor(Number(seconds)));
        const d = Math.floor(total / 86400);
        const h = Math.floor((total % 86400) / 3600);
        const m = Math.floor((total % 3600) / 60);
        const s = total % 60;
        if (d > 0) return `${d}d ${h}h`;
        if (h > 0) return `${h}h ${m}m`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
      },
      formatRelativeTime(iso) {
        if (!iso) return '';
        const date = new Date(iso);
        if (Number.isNaN(date.getTime())) return iso;
        const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
        if (diffSec < 60) return `${diffSec}s ago`;
        if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
        if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
        return `${Math.floor(diffSec / 86400)}d ago`;
      },
      prettyJobName(jobId) {
        const names = {
          post_tweet: 'Post Tweet',
          read_posts: 'Read Timeline',
          check_notifications: 'Check Notifications',
          process_inspiration_queue: 'Process Inspiration',
          process_replies: 'Process Replies',
        };
        return names[jobId] || String(jobId || '').replaceAll('_', ' ');
      },
      barStyle(value, max) {
        const safeMax = Math.max(Number(max) || 1, 1);
        const pct = Math.max(4, Math.min(100, Math.round(((Number(value) || 0) / safeMax) * 100)));
        return `width: ${pct}%`;
      },
      barHeightStyle(value, max) {
        const safeMax = Math.max(Number(max) || 1, 1);
        const pct = Math.max(8, Math.min(100, Math.round(((Number(value) || 0) / safeMax) * 100)));
        return `height: ${pct}%`;
      },
      async triggerJob(jobId) {
        if (!jobId) return;
        if (this.actionBusy[jobId]) return;
        this.actionBusy = { ...this.actionBusy, [jobId]: true };
        try {
          const response = await fetch(`/api/scheduler/run/${encodeURIComponent(jobId)}`, {
            method: 'POST',
            headers: { Accept: 'application/json' },
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || data.reason || 'Run failed');
          }
          await this.refreshOverview();
        } catch (error) {
          console.error(`Run-now failed for ${jobId}:`, error);
        } finally {
          this.actionBusy = { ...this.actionBusy, [jobId]: false };
        }
      },
    }));

    Alpine.data('settingsPage', (initialState) => ({
      topics: (initialState && initialState.topics) || [],
      topicInput: '',
      temperature: initialState && initialState.temperature != null ? initialState.temperature : 0.7,
      similarityThreshold:
        initialState && initialState.similarity_threshold != null
          ? initialState.similarity_threshold
          : 0.85,
      toastVisible: false,
      toastMessage: 'Settings saved',
      toastError: false,
      _toastTimer: null,
      init() {
        window.addEventListener('xbot:settings-save', () => this.saveAllSettings());
      },
      addTag(field, value) {
        const next = String(value || '').trim();
        if (!next || field !== 'topics') return;
        if (this.topics.includes(next)) return;
        this.topics.push(next);
        this.topicInput = '';
      },
      removeTag(field, value) {
        if (field !== 'topics') return;
        this.topics = this.topics.filter((item) => item !== value);
      },
      handleTagInput(event, field) {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        this.addTag(field, event.target.value);
      },
      updateRangeValue(input) {
        if (!input) return;
        const valueSpan = document.getElementById(`${input.id}-value`);
        if (valueSpan) {
          valueSpan.textContent = input.value;
        }
      },
      showToast(message, isError) {
        this.toastMessage = message;
        this.toastError = !!isError;
        this.toastVisible = true;
        if (this._toastTimer) {
          clearTimeout(this._toastTimer);
        }
        this._toastTimer = setTimeout(() => {
          this.toastVisible = false;
        }, 3000);
      },
      collectSettings() {
        return {
          personality: {
            tone: document.getElementById('tone').value,
            style: document.getElementById('style').value,
            topics: this.topics,
            min_tweet_length: parseInt(document.getElementById('min_tweet_length').value, 10),
            max_tweet_length: parseInt(document.getElementById('max_tweet_length').value, 10),
          },
          scheduler: {
            post_interval_hours: parseFloat(document.getElementById('post_interval_hours').value),
            post_jitter_hours: parseFloat(document.getElementById('post_jitter_hours').value),
            reading_check_minutes: parseInt(document.getElementById('reading_check_minutes').value, 10),
            mention_check_minutes: parseInt(document.getElementById('mention_check_minutes').value, 10),
            inspiration_check_minutes: parseInt(
              document.getElementById('inspiration_check_minutes').value,
              10,
            ),
            inspiration_batch_size: parseInt(
              document.getElementById('inspiration_batch_size').value,
              10,
            ),
          },
          llm: {
            provider: document.getElementById('provider').value,
            model: document.getElementById('model').value,
            max_tokens: parseInt(document.getElementById('max_tokens').value, 10),
            temperature: parseFloat(document.getElementById('temperature').value),
            embedding_provider: document.getElementById('embedding_provider').value,
            similarity_threshold: parseFloat(document.getElementById('similarity_threshold').value),
            recent_tweet_context_limit: parseInt(
              document.getElementById('recent_tweet_context_limit').value,
              10,
            ),
          },
          rate_limits: {
            max_posts_per_day: parseInt(document.getElementById('max_posts_per_day').value, 10),
            max_replies_per_day: parseInt(document.getElementById('max_replies_per_day').value, 10),
          },
          selenium: {
            min_delay_seconds: parseFloat(document.getElementById('min_delay_seconds').value),
            max_delay_seconds: parseFloat(document.getElementById('max_delay_seconds').value),
            headless: document.getElementById('headless').checked,
            user_agent_rotation: document.getElementById('user_agent_rotation').checked,
          },
        };
      },
      async saveAllSettings() {
        const btn = document.getElementById('save-all-btn');
        if (btn) {
          btn.disabled = true;
          btn.textContent = 'Saving...';
        }
        try {
          const settings = this.collectSettings();
          if (settings.personality.min_tweet_length > settings.personality.max_tweet_length) {
            throw new Error('Min Tweet Length must be <= Max Tweet Length');
          }
          const response = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
          });
          const result = await response.json();
          if (!response.ok) {
            throw new Error(result.detail || 'Failed to save settings');
          }
          this.showToast('Settings saved successfully', false);
        } catch (error) {
          this.showToast(`Error saving settings: ${error.message}`, true);
        } finally {
          if (btn) {
            btn.disabled = false;
            btn.textContent = 'Save All';
          }
        }
      },
    }));

    Alpine.data('chatPage', () => ({
      messages: [
        {
          role: 'assistant',
          text: "Hi! I'm ready to help draft content. Share your idea and I'll create options using the configured personality.",
          meta: 'Agent',
          allowPost: false,
          posted: false,
          showConfirm: false,
          posting: false,
          postError: '',
          streaming: false,
        },
      ],
      isGenerating: false,
      _renderRaf: null,
      init() {
        window.addEventListener('xbot:chat-clear', () => this.clearChat());
        const input = document.getElementById('chat-input');
        if (input) {
          input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              this.sendMessage();
            }
          });
        }
        this.renderMessages();
      },
      getEls() {
        return {
          form: document.getElementById('chat-form'),
          input: document.getElementById('chat-input'),
          windowEl: document.getElementById('chat-window'),
          statusEl: document.getElementById('chat-status'),
          sendBtn: document.getElementById('chat-send-btn'),
          toast: document.getElementById('chat-toast'),
        };
      },
      insertPrompt(prefix) {
        const { input } = this.getEls();
        if (!input) return;
        if (!input.value.trim()) {
          input.value = prefix;
        } else {
          input.value = `${input.value}\n${prefix}`;
        }
        input.focus();
      },
      showToast(message, isError) {
        const { toast } = this.getEls();
        if (!toast) return;
        const msgSpan = toast.querySelector('.toast-message');
        const iconSpan = toast.querySelector('.toast-icon');
        if (msgSpan) msgSpan.textContent = message;
        if (iconSpan) iconSpan.textContent = isError ? '✗' : '✓';
        toast.classList.toggle('toast-error', !!isError);
        toast.classList.remove('hidden');
        setTimeout(() => {
          toast.classList.add('hidden');
        }, 3000);
      },
      setStatus(text, isError) {
        const { statusEl } = this.getEls();
        const statusEl2 = document.getElementById('chat-status-secondary');
        if (!statusEl) return;
        statusEl.textContent = text || '';
        statusEl.classList.toggle('error', !!isError);
        if (statusEl2) {
          statusEl2.textContent = text || '';
          statusEl2.classList.toggle('error', !!isError);
        }
      },
      scheduleRender() {
        if (this._renderRaf) return;
        this._renderRaf = requestAnimationFrame(() => {
          this._renderRaf = null;
          this.renderMessages();
        });
      },
      renderMessages() {
        const { windowEl } = this.getEls();
        if (!windowEl) return;
        windowEl.innerHTML = '';
        this.messages.forEach((message, index) => {
          const {
            role,
            text,
            meta,
            allowPost,
            posted,
            posting,
            showConfirm,
            postError,
            postedAt,
            streaming,
          } = message;
          const msg = document.createElement('div');
          msg.className = `chat-message chat-message-${role}`;

          const avatar = document.createElement('div');
          avatar.className = 'chat-avatar';
          avatar.textContent = role === 'user' ? '👤' : '🤖';
          msg.appendChild(avatar);

          const bubble = document.createElement('div');
          bubble.className = 'chat-bubble';

          const textEl = document.createElement('div');
          textEl.className = 'chat-text';
          textEl.textContent = text;
          bubble.appendChild(textEl);

          if (streaming) {
            const cursor = document.createElement('span');
            cursor.className = 'chat-stream-cursor';
            cursor.textContent = '▋';
            textEl.appendChild(cursor);
          }

          if (meta) {
            const metaEl = document.createElement('div');
            metaEl.className = 'chat-info';
            metaEl.textContent = meta;
            bubble.appendChild(metaEl);
          } else if (role === 'assistant' && streaming) {
            const metaEl = document.createElement('div');
            metaEl.className = 'chat-info';
            metaEl.textContent = 'Thinking...';
            bubble.appendChild(metaEl);
          }

          if (role === 'assistant' && allowPost) {
            const actions = document.createElement('div');
            actions.className = 'chat-post-actions';

            const postBtn = document.createElement('button');
            postBtn.type = 'button';
            postBtn.className = 'btn btn-secondary btn-compact chat-post-btn';
            postBtn.textContent = posted ? 'Posted' : 'Post';
            postBtn.disabled = Boolean(posted) || Boolean(posting);

            const confirmEl = document.createElement('div');
            confirmEl.className = `chat-post-confirm${showConfirm ? '' : ' hidden'}`;

            const confirmText = document.createElement('span');
            confirmText.className = 'chat-post-confirm-text';
            confirmText.textContent = 'Post this message?';

            const confirmBtn = document.createElement('button');
            confirmBtn.type = 'button';
            confirmBtn.className = 'btn btn-primary btn-compact';
            confirmBtn.textContent = posting ? 'Posting...' : 'Confirm';
            confirmBtn.disabled = Boolean(posting);

            const cancelBtn = document.createElement('button');
            cancelBtn.type = 'button';
            cancelBtn.className = 'btn btn-secondary btn-compact';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.disabled = Boolean(posting);

            const statusEl = document.createElement('div');
            statusEl.className = 'chat-post-status';
            if (posting) {
              statusEl.textContent = 'Posting...';
            } else if (posted) {
              statusEl.textContent = `Posted ${postedAt ? new Date(postedAt).toLocaleTimeString() : ''}`;
              statusEl.classList.add('chat-post-success');
            } else if (postError) {
              statusEl.textContent = postError;
              statusEl.classList.add('chat-post-error');
            }

            postBtn.addEventListener('click', () => {
              if (posted || posting) return;
              this.messages[index].showConfirm = true;
              this.renderMessages();
            });
            confirmBtn.addEventListener('click', () => {
              if (posted || posting) return;
              this.postAssistantResponse(index);
            });
            cancelBtn.addEventListener('click', () => {
              this.messages[index].showConfirm = false;
              this.renderMessages();
            });

            confirmEl.append(confirmText, confirmBtn, cancelBtn);
            actions.appendChild(postBtn);
            actions.appendChild(confirmEl);
            if (statusEl.textContent) {
              actions.appendChild(statusEl);
            }
            bubble.appendChild(actions);
          }

          msg.appendChild(bubble);
          windowEl.appendChild(msg);
        });
        windowEl.scrollTop = windowEl.scrollHeight;
      },
      async postAssistantResponse(index) {
        const message = this.messages[index];
        if (!message || message.role !== 'assistant') return;
        this.messages[index].posting = true;
        this.messages[index].postError = '';
        this.setStatus('Posting...');
        this.renderMessages();
        try {
          const response = await fetch('/api/posts/manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: message.text }),
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || 'Post failed');
          }
          this.messages[index].posted = true;
          this.messages[index].postedAt = data.posted_at || new Date().toISOString();
          this.setStatus('');
          this.showToast('Tweet posted', false);
        } catch (error) {
          console.error('Manual post error', error);
          this.messages[index].postError = error.message || 'Post failed';
          this.setStatus(this.messages[index].postError, true);
          this.showToast(this.messages[index].postError, true);
        } finally {
          this.messages[index].posting = false;
          this.messages[index].showConfirm = false;
          this.renderMessages();
        }
      },
      async sendMessage(event) {
        if (event) event.preventDefault();
        const { input, sendBtn } = this.getEls();
        if (!input || !sendBtn) return;
        const content = input.value.trim();
        if (!content || this.isGenerating) return;

        this.messages.push({ role: 'user', text: content });
        const assistantIndex = this.messages.push({
          role: 'assistant',
          text: '',
          meta: '',
          allowPost: false,
          posted: false,
          posting: false,
          showConfirm: false,
          postError: '',
          streaming: true,
        }) - 1;
        this.renderMessages();
        this.setStatus('Generating response...', false);
        sendBtn.disabled = true;
        this.isGenerating = true;
        const messageToSend = content;
        input.value = '';
        input.focus();

        try {
          const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: messageToSend }),
          });
          if (!response.ok) {
            let detail = 'Chat failed';
            try {
              const data = await response.json();
              detail = data.detail || detail;
            } catch {
              // Ignore parse failure
            }
            throw new Error(detail);
          }

          const reader = response.body && response.body.getReader ? response.body.getReader() : null;
          if (!reader) {
            throw new Error('Streaming not supported in this browser');
          }

          const decoder = new TextDecoder();
          let buffer = '';
          let currentEvent = 'message';
          let streamError = null;

          const processEvent = (eventName, dataText) => {
            if (!dataText) return;
            let payload = {};
            try {
              payload = JSON.parse(dataText);
            } catch {
              payload = {};
            }

            if (eventName === 'chunk') {
              this.messages[assistantIndex].text += payload.delta || '';
              this.scheduleRender();
              this.setStatus('Generating response...', false);
            } else if (eventName === 'meta') {
              const provider = payload.provider || 'model';
              const totalTokens = payload.total_tokens ?? 0;
              this.messages[assistantIndex].meta = `${provider} · ${totalTokens} tokens`;
              this.messages[assistantIndex].allowPost = true;
              this.scheduleRender();
            } else if (eventName === 'error') {
              streamError = new Error(payload.error || 'Chat stream failed');
            }
          };

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let boundary = buffer.indexOf('\n\n');
            while (boundary !== -1) {
              const rawEvent = buffer.slice(0, boundary);
              buffer = buffer.slice(boundary + 2);
              let dataText = '';
              currentEvent = 'message';
              rawEvent.split('\n').forEach((line) => {
                if (line.startsWith('event:')) {
                  currentEvent = line.slice(6).trim();
                } else if (line.startsWith('data:')) {
                  dataText += line.slice(5).trim();
                }
              });
              processEvent(currentEvent, dataText);
              boundary = buffer.indexOf('\n\n');
            }
          }

          if (streamError) {
            throw streamError;
          }
          this.messages[assistantIndex].streaming = false;
          if (!this.messages[assistantIndex].meta) {
            this.messages[assistantIndex].meta = 'Agent';
          }
          this.setStatus('');
          this.showToast('Response received', false);
          this.renderMessages();
        } catch (error) {
          console.error('Chat error', error);
          if (this.messages[assistantIndex]) {
            this.messages[assistantIndex].streaming = false;
            if (!this.messages[assistantIndex].text) {
              this.messages.splice(assistantIndex, 1);
            } else {
              this.messages[assistantIndex].postError = error.message || 'Request failed';
              this.messages[assistantIndex].meta = 'Partial response';
            }
          }
          this.renderMessages();
          this.setStatus(error.message || 'Request failed', true);
          this.showToast(error.message || 'Request failed', true);
        } finally {
          this.isGenerating = false;
          sendBtn.disabled = false;
          input.focus();
        }
      },
      clearChat() {
        const { input } = this.getEls();
        this.messages.splice(0, this.messages.length, {
          role: 'assistant',
          text: 'Chat cleared. Share your next idea to begin.',
          meta: 'Agent',
          allowPost: false,
          posted: false,
          posting: false,
          showConfirm: false,
          streaming: false,
        });
        this.renderMessages();
        this.setStatus('');
        this.showToast('Chat cleared', false);
        if (input) input.focus();
      },
    }));

    Alpine.data('logsPage', () => ({
      evtSource: null,
      isPaused: false,
      statusText: 'Connecting…',
      statusType: 'info',
      init() {
        const level = document.getElementById('log-level');
        const tail = document.getElementById('tail-bytes');
        const reconnect = document.getElementById('reconnect-btn');
        const clear = document.getElementById('clear-btn');
        const pauseToggle = document.getElementById('pause-stream-toggle');
        const output = document.getElementById('log-output');
        const formatInputs = document.querySelectorAll('input[name="log-format"]');

        if (!level || !tail || !reconnect || !clear || !pauseToggle || !output) {
          return;
        }

        reconnect.addEventListener('click', () => this.connect());
        clear.addEventListener('click', () => {
          output.textContent = '';
        });
        level.addEventListener('change', () => this.connect());
        formatInputs.forEach((el) => el.addEventListener('change', () => this.connect()));
        pauseToggle.addEventListener('change', (e) => {
          this.isPaused = e.target.checked;
          this.setStatus(this.isPaused ? 'Paused' : 'Connected', this.isPaused ? 'warning' : 'success');
        });

        window.addEventListener('beforeunload', () => this.closeStream());
        this.connect();
      },
      setStatus(text, type) {
        this.statusText = text;
        this.statusType = type;
      },
      closeStream() {
        if (this.evtSource) {
          this.evtSource.close();
          this.evtSource = null;
        }
      },
      connect() {
        const output = document.getElementById('log-output');
        const level = document.getElementById('log-level');
        const tailBytes = document.getElementById('tail-bytes');
        const autoscroll = document.getElementById('autoscroll-toggle');
        const formatSelected = document.querySelector('input[name="log-format"]:checked');
        if (!output || !level || !tailBytes || !autoscroll) return;

        this.closeStream();
        output.textContent = '';

        const params = new URLSearchParams();
        if (level.value) {
          params.append('level', level.value);
        }
        params.append('tail_bytes', tailBytes.value || '0');
        params.append('human', formatSelected && formatSelected.value === 'human' ? 'true' : 'false');

        this.setStatus('Connecting…', 'info');
        this.evtSource = new EventSource(`/api/logs/stream?${params.toString()}`);

        this.evtSource.onopen = () => {
          this.setStatus('Connected', 'success');
        };

        this.evtSource.onerror = () => {
          this.setStatus('Disconnected', 'error');
        };

        this.evtSource.onmessage = (e) => {
          if (this.isPaused) return;
          const line = document.createElement('div');
          line.className = 'log-line';
          line.textContent = e.data;

          if (
            e.data.includes(' ERROR ') ||
            e.data.includes('"ERROR"') ||
            e.data.includes('❌ ')
          ) {
            line.classList.add('log-error');
          } else if (
            e.data.includes(' WARNING ') ||
            e.data.includes('"WARNING"') ||
            e.data.includes('⚠️ ')
          ) {
            line.classList.add('log-warning');
          } else if (
            e.data.includes(' DEBUG ') ||
            e.data.includes('"DEBUG"') ||
            e.data.includes('🔍 ')
          ) {
            line.classList.add('log-debug');
          }

          output.appendChild(line);
          while (output.childElementCount > 2000) {
            output.removeChild(output.firstChild);
          }
          if (autoscroll.checked) {
            output.scrollTop = output.scrollHeight;
          }
        };
      },
    }));

    Alpine.data('loginPage', () => ({
      init() {
        const userInput = document.getElementById('username');
        if (userInput) {
          userInput.focus();
        }
      },
      toggleTheme() {
        toggleTheme();
      },
    }));
  }

  document.addEventListener('alpine:init', () => {
    registerAlpineComponents(window.Alpine);
  });

  if (window.Alpine) {
    registerAlpineComponents(window.Alpine);
    try {
      if (document.body && typeof window.Alpine.initTree === 'function') {
        window.Alpine.initTree(document.body);
      }
    } catch (error) {
      console.warn('Alpine initTree retry failed', error);
    }
  }
})();
