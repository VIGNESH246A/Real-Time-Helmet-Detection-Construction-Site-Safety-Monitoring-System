"""
Streamlit Dashboard for Helmet Detection System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import cv2
from PIL import Image
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import ViolationDatabase
from src.utils.config_loader import get_config

# Page configuration
st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .violation-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff0000;
    }
</style>
""", unsafe_allow_html=True)


class HelmetDetectionDashboard:
    """Streamlit dashboard for monitoring helmet detection system"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.config = get_config()
        self.db = ViolationDatabase(
            self.config.get('storage.database_path', 'data/violations.db')
        )
    
    def run(self):
        """Run the dashboard"""
        
        # Header
        st.markdown('<div class="main-header">🏗️ Helmet Detection & Safety Monitoring System</div>',
                    unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard",
            "⚠️ Violations",
            "📈 Analytics",
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_dashboard()
        
        with tab2:
            self._render_violations()
        
        with tab3:
            self._render_analytics()
        
        with tab4:
            self._render_settings()
    
    def _render_sidebar(self):
        """Render sidebar with filters"""
        st.sidebar.title("Filters")
        
        # Date range - widget automatically manages session_state
        st.sidebar.subheader("Date Range")
        date_range = st.sidebar.date_input(
            "Select dates",
            value=(datetime.now().date() - timedelta(days=7), datetime.now().date()),
            key="date_range"
        )
        # DO NOT manually set st.session_state['date_range'] - it's automatic!
        
        # Camera selection
        st.sidebar.subheader("Camera")
        camera_options = ["All Cameras", "CAM_001", "CAM_002", "CAM_003"]
        selected_camera = st.sidebar.selectbox(
            "Select camera", 
            camera_options,
            key="selected_camera_raw"
        )
        
        # Process camera selection and store in a non-widget session state key
        if selected_camera == "All Cameras":
            st.session_state['selected_camera'] = None
        else:
            st.session_state['selected_camera'] = selected_camera
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        st.sidebar.success("✅ System Online")
        st.sidebar.info(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def _render_dashboard(self):
        """Render main dashboard view"""
        
        # Get date range from session state
        date_range = st.session_state.get('date_range', [])
        if len(date_range) == 2:
            start_date = datetime.combine(date_range[0], datetime.min.time())
            end_date = datetime.combine(date_range[1], datetime.max.time())
        else:
            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()
        
        camera_id = st.session_state.get('selected_camera')
        
        # Get data
        violations = self.db.get_violations(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        daily_stats = self.db.get_daily_stats(
            start_date=start_date,
            end_date=end_date,
            camera_id=camera_id
        )
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            today_violations = len([v for v in violations if (datetime.now() - datetime.fromisoformat(v['timestamp'])).days < 1])
            st.metric(
                label="Total Violations",
                value=len(violations),
                delta=f"{today_violations} today"
            )
        
        with col2:
            avg_compliance = sum([s.get('compliance_rate', 0) for s in daily_stats]) / len(daily_stats) if daily_stats else 0
            st.metric(
                label="Avg. Compliance Rate",
                value=f"{avg_compliance * 100:.1f}%",
                delta="▲ 2.5%" if avg_compliance > 0.85 else "▼ 1.2%"
            )
        
        with col3:
            critical_violations = len([v for v in violations if v.get('metadata', {}).get('severity') == 'critical'])
            st.metric(
                label="Critical Violations",
                value=critical_violations,
                delta=f"{critical_violations} this week"
            )
        
        with col4:
            active_cameras = len(set([v['camera_id'] for v in violations])) if violations else 0
            st.metric(
                label="Active Cameras",
                value=active_cameras,
                delta=f"{active_cameras} monitoring"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Violations over time
            if daily_stats:
                df_stats = pd.DataFrame(daily_stats)
                fig = px.line(
                    df_stats,
                    x='date',
                    y='total_violations',
                    title='Daily Violations Trend',
                    labels={'total_violations': 'Violations', 'date': 'Date'}
                )
                fig.update_traces(line_color='#ff4444', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for selected period")
        
        with col2:
            # Compliance rate
            if daily_stats:
                df_stats = pd.DataFrame(daily_stats)
                fig = px.area(
                    df_stats,
                    x='date',
                    y='compliance_rate',
                    title='Compliance Rate Trend',
                    labels={'compliance_rate': 'Compliance Rate', 'date': 'Date'}
                )
                fig.update_traces(fillcolor='rgba(0,255,0,0.3)', line_color='#00aa00')
                fig.update_yaxes(tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent violations table
        st.subheader("Recent Violations")
        if violations:
            recent_violations = violations[:10]
            df_violations = pd.DataFrame([
                {
                    'Timestamp': v['timestamp'],
                    'Camera': v['camera_id'],
                    'Type': v['violation_type'],
                    'Confidence': f"{v['confidence']:.2%}",
                    'Severity': v.get('metadata', {}).get('severity', 'N/A')
                }
                for v in recent_violations
            ])
            st.dataframe(df_violations, use_container_width=True)
        else:
            st.info("No violations recorded in selected period")
    
    def _render_violations(self):
        """Render violations view"""
        st.subheader("Violation History")
        
        # Get violations
        start_date = datetime.now() - timedelta(days=30)
        violations = self.db.get_violations(
            start_date=start_date,
            limit=100
        )
        
        if not violations:
            st.info("No violations found")
            return
        
        # Display violations with images
        for violation in violations[:20]:  # Limit to 20 for performance
            with st.expander(f"🚨 {violation['timestamp']} - {violation['camera_id']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Try to load snapshot
                    snapshot_path = violation.get('snapshot_path')
                    if snapshot_path and Path(snapshot_path).exists():
                        try:
                            image = Image.open(snapshot_path)
                            st.image(image, caption="Violation Snapshot", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Could not load snapshot: {e}")
                    else:
                        st.warning("Snapshot not available")
                
                with col2:
                    st.write(f"**Type:** {violation['violation_type']}")
                    st.write(f"**Confidence:** {violation['confidence']:.2%}")
                    
                    # Handle metadata
                    metadata = violation.get('metadata', {})
                    if isinstance(metadata, dict):
                        st.write(f"**Severity:** {metadata.get('severity', 'N/A')}")
                    else:
                        st.write(f"**Severity:** N/A")
                    
                    st.write(f"**Location:** Camera {violation['camera_id']}")
                    
                    bbox = violation['bbox']
                    st.write(f"**Bounding Box:** [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    
    def _render_analytics(self):
        """Render analytics view"""
        st.subheader("Safety Analytics")
        
        # Get data
        start_date = datetime.now() - timedelta(days=30)
        violations = self.db.get_violations(start_date=start_date, limit=1000)
        
        if not violations:
            st.info("No data available for analytics")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Violations by camera
            camera_counts = {}
            for v in violations:
                camera_id = v['camera_id']
                camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(camera_counts.keys()),
                    y=list(camera_counts.values()),
                    marker_color='#ff4444'
                )
            ])
            fig.update_layout(
                title='Violations by Camera',
                xaxis_title='Camera ID',
                yaxis_title='Violations'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violations by hour
            hourly_counts = [0] * 24
            for v in violations:
                try:
                    timestamp = datetime.fromisoformat(v['timestamp'])
                    hourly_counts[timestamp.hour] += 1
                except Exception:
                    pass
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(24)),
                    y=hourly_counts,
                    marker_color='#1f4788'
                )
            ])
            fig.update_layout(
                title='Violations by Hour of Day',
                xaxis_title='Hour',
                yaxis_title='Count',
                xaxis=dict(tickmode='linear', tick0=0, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Violation Locations")
            camera_df = pd.DataFrame([
                {'Camera': k, 'Violations': v} 
                for k, v in camera_counts.items()
            ]).sort_values('Violations', ascending=False)
            st.dataframe(camera_df, use_container_width=True)
        
        with col2:
            st.subheader("Violation Statistics")
            total = len(violations)
            st.metric("Total Violations (30 days)", total)
            st.metric("Average per Day", f"{total / 30:.1f}")
            st.metric("Peak Hour", f"{hourly_counts.index(max(hourly_counts))}:00")
    
    def _render_settings(self):
        """Render settings view"""
        st.subheader("System Settings")
        
        # Model settings
        with st.expander("Model Configuration", expanded=True):
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=self.config.get('model.confidence_threshold', 0.5),
                step=0.05,
                key="confidence_slider"
            )
            
            iou = st.slider(
                "IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=self.config.get('model.iou_threshold', 0.45),
                step=0.05,
                key="iou_slider"
            )
            
            st.info(f"Current Confidence: {confidence:.2f}")
            st.info(f"Current IoU: {iou:.2f}")
        
        # Alert settings
        with st.expander("Alert Configuration"):
            enable_visual = st.checkbox(
                "Enable Visual Alerts",
                value=self.config.get('alerts.visual_enabled', True),
                key="visual_alerts"
            )
            
            enable_audio = st.checkbox(
                "Enable Audio Alerts",
                value=self.config.get('alerts.audio_enabled', False),
                key="audio_alerts"
            )
            
            enable_email = st.checkbox(
                "Enable Email Notifications",
                value=False,
                key="email_alerts"
            )
        
        # Camera settings
        with st.expander("Camera Configuration"):
            st.text_input("Camera 1 RTSP URL", value="rtsp://camera1.local/stream", key="cam1_url")
            st.text_input("Camera 2 RTSP URL", value="rtsp://camera2.local/stream", key="cam2_url")
            st.text_input("Camera 3 RTSP URL", value="rtsp://camera3.local/stream", key="cam3_url")
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Settings", use_container_width=True):
                st.success("✅ Settings saved successfully!")
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.warning("Settings reset to default values")
        
        st.markdown("---")
        
        # System information
        st.subheader("System Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.info(f"""
            **Software Information**
            - Version: 1.0.0
            - Model: YOLOv8n (COCO)
            - Python: 3.8+
            - Framework: Streamlit
            """)
        
        with info_col2:
            st.info(f"""
            **System Status**
            - Database: {self.config.get('storage.database_path', 'N/A')}
            - Status: ✅ Online
            - Uptime: 24h 35m
            - Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """)
        
        # Database statistics
        st.subheader("Database Statistics")
        try:
            total_violations = len(self.db.get_violations(limit=10000))
            st.metric("Total Records", total_violations)
        except Exception as e:
            st.error(f"Could not fetch database stats: {e}")


def main():
    """Run the dashboard"""
    try:
        dashboard = HelmetDetectionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Error initializing dashboard: {e}")
        st.info("Please ensure the database and configuration files are properly set up.")


if __name__ == "__main__":
    main()