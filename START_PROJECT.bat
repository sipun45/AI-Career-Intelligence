@echo off
title AI Career Intelligence - Docker

cd /d C:\Users\Sipun\Desktop\ML-Career-Intelligence

echo ==========================================
echo    AI CAREER INTELLIGENCE
echo    Starting Docker Services...
echo ==========================================
echo.

docker compose up -d

echo.
echo ==========================================
echo    Docker Services Status
echo ==========================================
echo.

docker compose ps

echo.
echo ==========================================
echo    Project Started Successfully
echo ==========================================
echo.
echo Frontend: http://127.0.0.1:5173
echo API:      http://127.0.0.1:8000
echo.
echo Keep Docker Desktop running.
echo.

pause