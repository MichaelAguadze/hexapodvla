# Wi-Fi and SSH Guide

This is the most important setup guide in the repo.

The short version:

- Use the robot hotspot only for first access.
- Move the robot onto the same Wi-Fi as your laptop as soon as possible.
- Once both devices share one network, SSH, recording, and file transfer become much easier.

## Two Network Modes

### AP mode

In AP mode, the robot creates its own hotspot. This is useful when you first power it on because you can always get into the robot even if you do not know any LAN IP yet.

Typical defaults on TurboPi Advanced images:

- Hotspot name starts with `HW`
- Hotspot password is `hiwonder`
- Robot IP is `192.168.149.1`

### Shared Wi-Fi mode

In shared Wi-Fi mode, the robot joins the same router or hotspot as your laptop.

This is the mode you want for normal development because:

- the laptop keeps internet access
- the robot and laptop can reach each other normally
- you do not need to keep switching Wi-Fi networks during a session

## First-Time Setup With nmcli

### 1. Connect to the robot hotspot

1. Power on the robot.
2. Wait for the boot sequence to finish.
3. Join the Wi-Fi network whose name starts with `HW`.

### 2. SSH into the robot

```bash
ssh pi@192.168.149.1
```

The username is usually `pi`. On many stock images, the password is `raspberrypi`.

### 3. List nearby Wi-Fi networks

Once you are inside the robot:

```bash
nmcli dev wifi list
```

Pick the SSID that your laptop will also use.

### 4. Join the shared Wi-Fi

```bash
sudo nmcli device wifi connect "<SSID>" password "<PASSWORD>"
```

What happens next:

- the current SSH session may disconnect immediately
- the robot drops the hotspot network because it is joining another network
- this is normal and expected

### 5. Reconnect your laptop to the same Wi-Fi

After the robot joins the shared network, connect your laptop to that same Wi-Fi network.

### 6. Find the new robot IP

Use one of these methods:

- Router admin page or DHCP client list
- WonderPi app or other vendor tooling
- A local keyboard and display on the robot, then run `hostname -I`

### 7. SSH back in

```bash
ssh pi@<ROBOT_IP>
```

At this point, you are in the best workflow for this repo.

## Make It Persistent With wifi_conf.py

After the first successful `nmcli` connection, you can make the shared Wi-Fi configuration survive reboots.

On the robot:

```bash
cd ~/hiwonder-toolbox
sudo nano wifi_conf.py
```

Many TurboPi Advanced images expose settings similar to:

```python
HW_WIFI_MODE = 2
HW_WIFI_STA_SSID = "<SSID>"
HW_WIFI_STA_PASSWORD = "<PASSWORD>"
```

Then restart the Wi-Fi service:

```bash
sudo systemctl restart wifi.service
```

Notes:

- Some images use slightly different constant names in `wifi_conf.py`
- If your file looks different, update the mode, SSID, and password fields that the image already provides
- If the configured Wi-Fi is unavailable, the robot may fall back to hotspot mode so you do not get locked out

## Reboot Recovery

If the robot comes back on the hotspot after a reboot:

1. Rejoin the `HW` hotspot.
2. SSH to `192.168.149.1`.
3. Either run the `nmcli` connection command again or fix the persistent `wifi_conf.py` settings.

## Why We Do It This Way

Students often ask why we do not stay on the robot hotspot forever.

The answer is simple:

- hotspot mode is good for first access
- shared Wi-Fi is better for everyday work

Once both devices are on the same network, you can SSH, install packages, start the server, run the client, and save data without constantly switching Wi-Fi connections.

## References

- [Hiwonder TurboPi network setup](https://docs.hiwonder.com/projects/TurboPi/en/advanced/docs/7.network_configuration.html)
- [Hiwonder TurboPi getting ready](https://docs.hiwonder.com/projects/TurboPi/en/latest/docs/1.getting_ready.html)
