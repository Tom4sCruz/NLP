@echo off
REM ============================================================
REM  testing_all.bat  (Windows)
REM  Runs RNN and LinearSVC experiments for the given split ratio
REM  Usage:
REM     testing_all.bat 0.8
REM ============================================================

REM --- Get the absolute path of this .bat file
set "SCRIPT_DIR=%~dp0"

REM --- Move to the parent Scripts folder
for %%I in ("%SCRIPT_DIR%..") do set "SCRIPTS_DIR=%%~fI"
cd /d "%SCRIPTS_DIR%"

echo.
echo ============================================================
echo [INFO] Now running from: %CD%
echo ============================================================
echo.

REM --- Check if split ratio is passed
if "%~1"=="" (
    echo Usage: testing_all.bat ^<split_ratio^>
    echo Example: testing_all.bat 0.8
    pause
    exit /b 1
)

set "SPLIT=%~1"

REM --- Define paths (relative to Scripts/)
set "TRAIN_PATH=..\Datasets\Splits\training_set_%SPLIT%.csv"
set "TEST_PATH=..\Datasets\Splits\testing_set_%SPLIT%.csv"
set "TEST_PATH_NO_CHEF=..\Datasets\Splits\testing_set_%SPLIT%_no_chef_id.csv"

set "RESULT_RNN=..\Results\results_rnn_testing_set_%SPLIT%_no_chef_id.txt"
set "RESULT_SCV=..\Results\results_linearSVC_testing_set_%SPLIT%_no_chef_id.txt"

set "HIDDEN_DIMS=128"
set "EPOCHS_LIST=30"

REM --- Ensure Results directory exists
if not exist "..\Results" (
    mkdir "..\Results"
)

REM ============================================================
REM Run RNN and LinearSVC Experiments
REM ============================================================

for %%H in (%HIDDEN_DIMS%) do (
    for %%E in (%EPOCHS_LIST%) do (
        echo [+] Running RNN with HIDDEN_DIM=%%H ^| EPOCHS=%%E ^| MAX_LEN=100 ^| SPLIT_RATIO=%SPLIT%
        echo ------------------------------------------------------------

        python "Models\rnn.py" "%TRAIN_PATH%" "%TEST_PATH_NO_CHEF%" --hidden_dim %%H --epochs %%E --max_len 100
        if errorlevel 1 (
            echo [!] Error running RNN script.
            pause
            exit /b 1
        )

        python "Misc\obtain_metrics.py" "%RESULT_RNN%" "%TEST_PATH%"
        if errorlevel 1 (
            echo [!] Error running obtain_metrics.
            pause
            exit /b 1
        )

        echo [+] Finished run: H=%%H ^| E=%%E ^| L=100 ^| SPLIT_RATIO=%SPLIT%
        echo ------------------------------------------------------------
    )
)

python "Models\linearSCV.py" "%TRAIN_PATH%" "%TEST_PATH_NO_CHEF%"
python "Misc\obtain_metrics.py" "%RESULT_SCV%" "%TEST_PATH%"

echo.
echo [✅] All experiments completed successfully!
pause
