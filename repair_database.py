#!/usr/bin/env python
"""
Utility script to repair a locked or corrupted SQLite database.
Run this script if you experience "database is locked" errors.
"""
import os
import sqlite3
import shutil
import time
from pathlib import Path

def repair_database(db_path: str):
    """Check and repair the SQLite database"""
    print(f"Checking database at {db_path}...")
    
    # Ensure the database file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return False
    
    # Create a backup copy
    backup_path = f"{db_path}.backup_{int(time.time())}"
    print(f"Creating backup at {backup_path}...")
    shutil.copy2(db_path, backup_path)
    
    try:
        # Try to open the database and run PRAGMA integrity_check
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        
        print("Running integrity check...")
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        if result[0] == "ok":
            print("Database integrity check passed.")
        else:
            print(f"Database integrity issues found: {result[0]}")
            print("Attempting to repair...")
            
            # Run vacuum to rebuild the database
            print("Running VACUUM...")
            cursor.execute("VACUUM")
            conn.commit()
            
        # Optimize the database
        print("Optimizing database...")
        cursor.execute("PRAGMA optimize")
        
        # Check for locks
        cursor.execute("PRAGMA lock_status")
        locks = cursor.fetchall()
        print(f"Lock status: {locks}")
        
        # Close connection properly
        cursor.close()
        conn.close()
        
        print("Database check and optimization completed!")
        return True
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        
        # If we couldn't repair it, restore the backup
        print(f"Restoring from backup {backup_path}...")
        shutil.copy2(backup_path, db_path)
        
        return False

if __name__ == "__main__":
    # Get the database path from Django settings
    import os
    import django
    
    # Setup Django environment
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "faster_chat.settings")
    django.setup()
    
    from django.conf import settings
    
    db_path = settings.DATABASES['default']['NAME']
    repair_database(db_path) 