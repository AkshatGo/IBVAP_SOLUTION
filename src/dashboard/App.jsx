/**
 * IBVAP Dashboard
 * React-based real-time surveillance dashboard
 */

import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polygon } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// ============================================================
// Mock Data (replace with API calls in production)
// ============================================================

const MOCK_SITES = [
  {
    site_id: 'BOP-01',
    site_name: 'Border Post Alpha',
    latitude: 34.0837,
    longitude: 74.7973,
    status: 'online',
    cameras: [
      { camera_id: 'CAM-01', status: 'online', type: 'fixed' },
      { camera_id: 'CAM-02', status: 'online', type: 'ptz' },
      { camera_id: 'CAM-03', status: 'offline', type: 'fixed' },
    ],
    alert_count: 12,
  },
  {
    site_id: 'BOP-02',
    site_name: 'Border Post Bravo',
    latitude: 34.1234,
    longitude: 74.8123,
    status: 'online',
    cameras: [
      { camera_id: 'CAM-01', status: 'online', type: 'fixed' },
      { camera_id: 'CAM-02', status: 'online', type: 'fixed' },
    ],
    alert_count: 5,
  },
  {
    site_id: 'BOP-03',
    site_name: 'Border Post Charlie',
    latitude: 34.0512,
    longitude: 74.7645,
    status: 'degraded',
    cameras: [
      { camera_id: 'CAM-01', status: 'online', type: 'fixed' },
    ],
    alert_count: 0,
  },
];

const MOCK_EVENTS = [
  {
    event_id: 'e001',
    timestamp: new Date(Date.now() - 300000).toISOString(),
    site_id: 'BOP-01',
    camera_id: 'CAM-01',
    event_type: 'fence_intrusion',
    object_class: 'person',
    track_id: 'T-0042',
    zone: 'Zone-3',
    confidence: 0.91,
    explanation: 'Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE',
    severity: 'high',
  },
  {
    event_id: 'e002',
    timestamp: new Date(Date.now() - 600000).toISOString(),
    site_id: 'BOP-01',
    camera_id: 'CAM-02',
    event_type: 'anpr_match',
    object_class: 'vehicle',
    track_id: 'T-0015',
    zone: 'Checkpoint-1',
    confidence: 0.87,
    explanation: 'Vehicle BR12AB3456 detected at Checkpoint-1',
    severity: 'medium',
  },
  {
    event_id: 'e003',
    timestamp: new Date(Date.now() - 900000).toISOString(),
    site_id: 'BOP-02',
    camera_id: 'CAM-01',
    event_type: 'signal_loss',
    object_class: 'none',
    track_id: 'N/A',
    zone: 'N/A',
    confidence: 1.0,
    explanation: 'Camera CAM-01 signal lost for 5 seconds',
    severity: 'critical',
  },
];

// ============================================================
// Components
// ============================================================

const SeverityBadge = ({ severity }) => {
  const colors = {
    low: '#4caf50',
    medium: '#ff9800',
    high: '#f44336',
    critical: '#9c27b0',
  };

  return (
    <span
      style={{
        backgroundColor: colors[severity] || '#757575',
        color: 'white',
        padding: '2px 8px',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: 'bold',
      }}
    >
      {severity.toUpperCase()}
    </span>
  );
};

const StatusIndicator = ({ status }) => {
  const colors = {
    online: '#4caf50',
    offline: '#f44336',
    degraded: '#ff9800',
  };

  return (
    <span
      style={{
        display: 'inline-block',
        width: '10px',
        height: '10px',
        borderRadius: '50%',
        backgroundColor: colors[status] || '#757575',
        marginRight: '5px',
      }}
    />
  );
};

const AlertCard = ({ event, onClick }) => (
  <div
    onClick={() => onClick(event)}
    style={{
      backgroundColor: '#1e1e2e',
      borderRadius: '8px',
      padding: '12px',
      marginBottom: '8px',
      cursor: 'pointer',
      borderLeft: `4px solid ${
        event.severity === 'critical'
          ? '#9c27b0'
          : event.severity === 'high'
          ? '#f44336'
          : event.severity === 'medium'
          ? '#ff9800'
          : '#4caf50'
      }`,
    }}
  >
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
      <span style={{ color: '#aaa', fontSize: '12px' }}>
        {new Date(event.timestamp).toLocaleTimeString()}
      </span>
      <SeverityBadge severity={event.severity} />
    </div>
    <div style={{ color: 'white', fontWeight: 'bold', marginBottom: '4px' }}>
      {event.event_type.replace(/_/g, ' ').toUpperCase()}
    </div>
    <div style={{ color: '#aaa', fontSize: '12px' }}>
      {event.site_id} • {event.camera_id} • {event.zone}
    </div>
    <div style={{ color: '#888', fontSize: '11px', marginTop: '8px' }}>
      {event.explanation}
    </div>
  </div>
);

const SiteCard = ({ site }) => (
  <div
    style={{
      backgroundColor: '#1e1e2e',
      borderRadius: '8px',
      padding: '16px',
      marginBottom: '12px',
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
      <StatusIndicator status={site.status} />
      <span style={{ color: 'white', fontWeight: 'bold' }}>{site.site_name}</span>
    </div>
    <div style={{ color: '#aaa', fontSize: '12px', marginBottom: '8px' }}>
      {site.site_id} • {site.cameras.length} cameras
    </div>
    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
      {site.cameras.map((cam) => (
        <div
          key={cam.camera_id}
          style={{
            backgroundColor: cam.status === 'online' ? '#2d4a3e' : '#4a2d2d',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
          }}
        >
          <StatusIndicator status={cam.status} />
          <span style={{ color: '#ccc' }}>{cam.camera_id}</span>
        </div>
      ))}
    </div>
    <div style={{ color: '#ff9800', fontSize: '12px', marginTop: '8px' }}>
      {site.alert_count} alerts today
    </div>
  </div>
);

const StatsCard = ({ title, value, icon, color }) => (
  <div
    style={{
      backgroundColor: '#1e1e2e',
      borderRadius: '8px',
      padding: '16px',
      textAlign: 'center',
    }}
  >
    <div style={{ fontSize: '24px', marginBottom: '8px' }}>{icon}</div>
    <div style={{ color: 'white', fontSize: '28px', fontWeight: 'bold' }}>{value}</div>
    <div style={{ color: '#aaa', fontSize: '12px' }}>{title}</div>
  </div>
);

// ============================================================
// Main Dashboard
// ============================================================

const Dashboard = () => {
  const [sites, setSites] = useState(MOCK_SITES);
  const [events, setEvents] = useState(MOCK_EVENTS);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [activeTab, setActiveTab] = useState('alerts');
  const [stats, setStats] = useState({
    total_sites: MOCK_SITES.length,
    online_sites: MOCK_SITES.filter((s) => s.status === 'online').length,
    total_cameras: MOCK_SITES.reduce((acc, s) => acc + s.cameras.length, 0),
    online_cameras: MOCK_SITES.reduce(
      (acc, s) => acc + s.cameras.filter((c) => c.status === 'online').length,
      0
    ),
    total_alerts_today: events.length,
    critical_alerts: events.filter((e) => e.severity === 'critical').length,
    high_alerts: events.filter((e) => e.severity === 'high').length,
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Add a new random event
      const newEvent = {
        event_id: `e${Date.now()}`,
        timestamp: new Date().toISOString(),
        site_id: MOCK_SITES[Math.floor(Math.random() * MOCK_SITES.length)].site_id,
        camera_id: `CAM-0${Math.floor(Math.random() * 3) + 1}`,
        event_type: ['fence_intrusion', 'vehicle_detected', 'person_detected'][
          Math.floor(Math.random() * 3)
        ],
        object_class: Math.random() > 0.5 ? 'person' : 'vehicle',
        track_id: `T-${Math.floor(Math.random() * 1000)}`,
        zone: `Zone-${Math.floor(Math.random() * 5) + 1}`,
        confidence: 0.7 + Math.random() * 0.3,
        explanation: 'Simulated event for demo purposes',
        severity: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
      };

      setEvents((prev) => [newEvent, ...prev.slice(0, 49)]); // Keep last 50
      setStats((prev) => ({
        ...prev,
        total_alerts_today: prev.total_alerts_today + 1,
      }));
    }, 10000); // New event every 10 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        backgroundColor: '#0d1117',
        minHeight: '100vh',
        color: 'white',
        fontFamily: 'Inter, -apple-system, sans-serif',
      }}
    >
      {/* Header */}
      <header
        style={{
          backgroundColor: '#161b22',
          padding: '16px 24px',
          borderBottom: '1px solid #30363d',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '24px' }}>🛡️</span>
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>IBVAP</h1>
            <p style={{ margin: 0, fontSize: '12px', color: '#8b949e' }}>
              Intelligent Border Video Analytics Platform
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <StatusIndicator status="online" />
          <span style={{ color: '#8b949e', fontSize: '12px' }}>System Online</span>
          <span style={{ color: '#8b949e', fontSize: '12px' }}>
            {new Date().toLocaleString()}
          </span>
        </div>
      </header>

      {/* Stats Bar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(6, 1fr)',
          gap: '16px',
          padding: '16px 24px',
          borderBottom: '1px solid #30363d',
        }}
      >
        <StatsCard title="Sites" value={stats.total_sites} icon="🏢" color="#4caf50" />
        <StatsCard
          title="Online Sites"
          value={stats.online_sites}
          icon="✅"
          color="#4caf50"
        />
        <StatsCard title="Cameras" value={stats.total_cameras} icon="📷" color="#2196f3" />
        <StatsCard
          title="Online Cameras"
          value={stats.online_cameras}
          icon="🟢"
          color="#4caf50"
        />
        <StatsCard
          title="Alerts Today"
          value={stats.total_alerts_today}
          icon="🔔"
          color="#ff9800"
        />
        <StatsCard
          title="Critical Alerts"
          value={stats.critical_alerts}
          icon="🚨"
          color="#f44336"
        />
      </div>

      {/* Main Content */}
      <div style={{ display: 'flex', height: 'calc(100vh - 200px)' }}>
        {/* Map */}
        <div style={{ flex: 1, position: 'relative' }}>
          <MapContainer
            center={[34.0837, 74.7973]}
            zoom={10}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {sites.map((site) => (
              <Marker key={site.site_id} position={[site.latitude, site.longitude]}>
                <Popup>
                  <div>
                    <strong>{site.site_name}</strong>
                    <br />
                    {site.cameras.length} cameras • {site.alert_count} alerts
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>

        {/* Sidebar */}
        <div
          style={{
            width: '350px',
            backgroundColor: '#161b22',
            borderLeft: '1px solid #30363d',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Tabs */}
          <div
            style={{
              display: 'flex',
              borderBottom: '1px solid #30363d',
            }}
          >
            {['alerts', 'sites', 'cameras'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: activeTab === tab ? '#1e1e2e' : 'transparent',
                  color: activeTab === tab ? 'white' : '#8b949e',
                  border: 'none',
                  cursor: 'pointer',
                  fontWeight: activeTab === tab ? 'bold' : 'normal',
                  borderBottom: activeTab === tab ? '2px solid #58a6ff' : 'none',
                }}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
            {activeTab === 'alerts' &&
              events.map((event) => (
                <AlertCard
                  key={event.event_id}
                  event={event}
                  onClick={setSelectedEvent}
                />
              ))}

            {activeTab === 'sites' &&
              sites.map((site) => <SiteCard key={site.site_id} site={site} />)}

            {activeTab === 'cameras' && (
              <div style={{ color: '#8b949e', textAlign: 'center', padding: '24px' }}>
                Camera management coming soon
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Event Detail Modal */}
      {selectedEvent && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setSelectedEvent(null)}
        >
          <div
            style={{
              backgroundColor: '#1e1e2e',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '500px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <h2 style={{ margin: 0 }}>Event Details</h2>
              <button
                onClick={() => setSelectedEvent(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#8b949e',
                  cursor: 'pointer',
                  fontSize: '20px',
                }}
              >
                ×
              </button>
            </div>

            <div style={{ marginBottom: '12px' }}>
              <SeverityBadge severity={selectedEvent.severity} />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ color: '#8b949e', fontSize: '12px' }}>Event Type</div>
              <div style={{ color: 'white' }}>
                {selectedEvent.event_type.replace(/_/g, ' ').toUpperCase()}
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ color: '#8b949e', fontSize: '12px' }}>Explanation</div>
              <div style={{ color: 'white' }}>{selectedEvent.explanation}</div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                marginBottom: '16px',
              }}
            >
              <div>
                <div style={{ color: '#8b949e', fontSize: '12px' }}>Site</div>
                <div style={{ color: 'white' }}>{selectedEvent.site_id}</div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: '12px' }}>Camera</div>
                <div style={{ color: 'white' }}>{selectedEvent.camera_id}</div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: '12px' }}>Zone</div>
                <div style={{ color: 'white' }}>{selectedEvent.zone}</div>
              </div>
              <div>
                <div style={{ color: '#8b949e', fontSize: '12px' }}>Confidence</div>
                <div style={{ color: 'white' }}>{(selectedEvent.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <div style={{ color: '#8b949e', fontSize: '12px' }}>Timestamp</div>
              <div style={{ color: 'white' }}>
                {new Date(selectedEvent.timestamp).toLocaleString()}
              </div>
            </div>

            <button
              style={{
                width: '100%',
                padding: '12px',
                backgroundColor: '#238636',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: 'bold',
              }}
            >
              View Clip
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
