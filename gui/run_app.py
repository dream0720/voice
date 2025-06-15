#!/usr/bin/env python3
"""
Voice Processing Suite - Application Launcher
==============================================

Modern voice processing application with improved UI and functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main application entry point"""
    try:
        # Import and run the application
        from gui.voice_processing_app import main as run_app
        run_app()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("请确保已安装所有必需的依赖包。")
        print("运行: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
