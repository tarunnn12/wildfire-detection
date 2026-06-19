const els = {
  badge:      document.getElementById('system-badge'),
  badgeText:  document.getElementById('system-status-text'),
  pulseDot:   document.getElementById('pulse-dot'),
  statusCard: document.getElementById('status-card'),
  iconWrap:   document.getElementById('status-icon-wrap'),
  iconClear:  document.getElementById('icon-clear'),
  iconFire:   document.getElementById('icon-fire'),
  iconSmoke:  document.getElementById('icon-smoke'),
  statusLabel:document.getElementById('status-label'),
  statusSub:  document.getElementById('status-sub'),
  detList:    document.getElementById('det-list'),
  statFrames: document.getElementById('stat-frames'),
  statFire:   document.getElementById('stat-fire'),
  statSmoke:  document.getElementById('stat-smoke'),
  statAlert:  document.getElementById('stat-alert'),
  fpsDisplay: document.getElementById('fps-display'),
  feedTime:   document.getElementById('feed-time'),
  alertOverlay: document.getElementById('alert-overlay'),
  alertBanner:  document.getElementById('alert-banner'),
  alertText:    document.getElementById('alert-text'),
};

function setStatus(status) {
  const isFireState  = status === 'fire';
  const isSmokeState = status === 'smoke';
  const isClear      = status === 'clear' || status === 'monitoring';

  // Header badge
  els.badge.className = 'header-badge' + (isFireState ? ' fire' : isSmokeState ? ' smoke' : '');
  els.badgeText.textContent = isFireState ? 'Fire detected' : isSmokeState ? 'Smoke detected' : 'Monitoring';

  // Status card
  els.statusCard.className = 'card status-card' + (isFireState ? ' fire' : isSmokeState ? ' smoke' : '');
  els.iconWrap.className   = 'status-icon-wrap'  + (isFireState ? ' fire' : isSmokeState ? ' smoke' : '');

  // Icons
  els.iconClear.style.display = isClear      ? 'block' : 'none';
  els.iconFire.style.display  = isFireState  ? 'block' : 'none';
  els.iconSmoke.style.display = isSmokeState ? 'block' : 'none';

  // Label
  els.statusLabel.className   = 'status-label' + (isFireState ? ' fire' : isSmokeState ? ' smoke' : '');
  els.statusLabel.textContent = isFireState ? 'Fire detected' : isSmokeState ? 'Smoke detected' : 'All clear';
  els.statusSub.textContent   = isFireState ? 'Immediate threat — take action'
                               : isSmokeState ? 'Smoke presence detected'
                               : 'No threats detected';

  // Alert overlay
  if (isFireState) {
    els.alertOverlay.classList.add('visible');
    els.alertBanner.className = 'alert-banner';
    els.alertText.textContent = 'FIRE DETECTED';
  } else if (isSmokeState) {
    els.alertOverlay.classList.add('visible');
    els.alertBanner.className = 'alert-banner smoke-alert';
    els.alertText.textContent = 'SMOKE DETECTED';
  } else {
    els.alertOverlay.classList.remove('visible');
  }
}

function renderDetections(dets) {
  if (!dets || dets.length === 0) {
    els.detList.innerHTML = '<li class="det-empty">No active detections</li>';
    return;
  }
  els.detList.innerHTML = dets.map(d => `
    <li class="det-item ${d.class}">
      <span class="det-name">${d.class}</span>
      <span class="det-conf">${Math.round(d.conf * 100)}% conf</span>
    </li>`).join('');
}

function updateClock() {
  const now = new Date();
  els.feedTime.textContent = now.toLocaleTimeString('en-AU', { hour12: false });
}

async function poll() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();

    setStatus(data.status);
    renderDetections(data.detections);

    els.statFrames.textContent = data.frame_count.toLocaleString();
    els.statFire.textContent   = data.fire_count.toLocaleString();
    els.statSmoke.textContent  = data.smoke_count.toLocaleString();
    els.statAlert.textContent  = data.last_alert || '--';
    els.fpsDisplay.textContent = data.fps + ' fps';
  } catch (e) {
    // silently skip on network error
  }
}

async function sendManualAlert() {
  const btn = document.getElementById('send-alert-btn');
  btn.disabled = true;
  btn.textContent = 'SENDING...';

  try {
    const res = await fetch('/api/send_alert', { method: 'POST' });
    const data = await res.json();

    if (data.sent) {
      btn.textContent = '✓ ALERT SENT';
      btn.style.background = 'var(--clear)';
    } else {
      btn.textContent = data.message;
      btn.style.background = 'var(--text-muted)';
    }
  } catch (e) {
    btn.textContent = 'FAILED — RETRY';
  }

  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = 'SEND ALERT';
    btn.style.background = 'var(--fire)';
  }, 3000);
}

setInterval(poll, 700);
setInterval(updateClock, 1000);
updateClock();
poll();
