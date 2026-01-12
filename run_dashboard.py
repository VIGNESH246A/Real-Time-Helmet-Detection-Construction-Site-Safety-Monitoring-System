"""
Dashboard launcher with correct Python path
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run dashboard
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import os
    
    dashboard_path = project_root / "src" / "ui" / "dashboard.py"
    
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main())