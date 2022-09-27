# Runs the rave-web suite calling each script individually
start powershell { npm run --prefix ./rave-server dev; Read-Host }
start powershell { npm run --prefix ./rave-web start; Read-Host }
start powershell { cd ./library/RAVE/src; python main_vision_web.py; Read-Host }