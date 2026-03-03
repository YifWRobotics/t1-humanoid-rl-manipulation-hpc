#!/bin/bash
set -e

echo "stopping docker services..."
sudo systemctl stop docker.service docker.socket containerd.service 2>/dev/null || true

echo "unholding docker packages..."
sudo apt-mark unhold docker-ce docker-ce-cli 2>/dev/null || true

echo "removing docker packages..."
sudo apt-get purge -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>/dev/null || true
sudo apt autoremove -y

echo "cleaning docker directories..."
sudo umount /var/lib/docker/overlay2/*/merged 2>/dev/null || true
sudo rm -rf /var/lib/docker /var/lib/containerd

echo "uninstalling current apptainer..."
sudo apt-get purge -y apptainer 2>/dev/null || true
sudo apt autoremove -y
sudo rm -rf /usr/local/bin/apptainer /usr/local/etc/apptainer

echo "installing docker 24.0.7..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update

# check available versions and install
echo "checking available docker versions..."
apt-cache madison docker-ce | grep 24.0.7 | head -1

# install available 24.0.7 version
UBUNTU_VERSION=$(lsb_release -cs)
sudo apt-get install -y docker-ce=5:24.0.7-1~ubuntu.22.04~${UBUNTU_VERSION} docker-ce-cli=5:24.0.7-1~ubuntu.22.04~${UBUNTU_VERSION} containerd.io

sudo apt-mark hold docker-ce docker-ce-cli

echo "installing apptainer 1.2.5..."
wget -q https://github.com/apptainer/apptainer/releases/download/v1.2.5/apptainer_1.2.5_amd64.deb
sudo dpkg -i apptainer_1.2.5_amd64.deb
rm apptainer_1.2.5_amd64.deb

echo "verification:"
docker --version
apptainer --version

echo "adding user to docker group..."
sudo usermod -aG docker $USER
echo "logout/login required for docker group changes"