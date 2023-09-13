@echo off

:start
cls
set mypath=%cd%
@echo %mypath%

pip install virtualenv
virtualenv venv

cd mypath
call venv\Scripts\activate.bat

pip install -r requirements.txt

start "" http://127.0.0.1:5000/

python app.py

@echo Starting server and Opening browser...
timeout /t 2 /nobreak

pause
exit
