@echo off
echo ======================================================
echo Fixing database lock issues and restarting the server
echo ======================================================

echo Stopping all Python processes...
taskkill /F /IM python.exe 2>nul

echo.
echo Waiting for processes to terminate...
timeout /t 2 /nobreak >nul

echo.
echo Repairing database...
python repair_database.py

echo.
echo Setting proper permissions on database file...
icacls db.sqlite3 /grant Everyone:F

echo.
echo Clearing system-wide OPENAI_API_KEY environment variable...
set "OPENAI_API_KEY="

echo.
echo ======================================================
echo Starting Django server with a clean environment...
echo ======================================================
python manage.py runserver 