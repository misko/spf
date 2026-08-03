sudo systemctl stop mavlink_controller.service # might fail but just in case
sleep 5
wget https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/scripts/uploader.py
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv3/ardurover.apj
wget https://firmware.ardupilot.org/Rover/stable-4.5.0/fmuv3/ardurover.apj
#wget https://firmware.ardupilot.org/Rover/stable-4.4.0/fmuv2/ardurover.apj
# uploader.py needs pyserial, which is in the SPF virtualenv but not in the
# system python on a Raspberry Pi OS Lite image. Plain `python` therefore fails
# with "ModuleNotFoundError: No module named 'serial'" on a freshly provisioned
# rover (hit on Rover 4). Prefer the venv interpreter and fall back only if it
# is missing.
SPF_VENV_PYTHON="${SPF_VENV_PYTHON:-/home/pi/spf-virtualenv/bin/python}"
if [ -x "$SPF_VENV_PYTHON" ]; then
    uploader_python="$SPF_VENV_PYTHON"
else
    uploader_python="$(command -v python3 || command -v python)"
fi
echo "flash_ardupilot: using ${uploader_python}"
"$uploader_python" uploader.py ardurover.apj 2>&1 | tee ardurover_flash.log
sleep 5