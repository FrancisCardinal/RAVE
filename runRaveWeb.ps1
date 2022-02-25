# Runs the rave-web suite calling each script individually
start powershell { npm run --prefix ./rave-server dev; Read-Host }
start powershell { npm run --prefix ./rave-web start; Read-Host }
start powershell { cd ./library/RAVE/src; python moc_tracking_client.py; Read-Host }