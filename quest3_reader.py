"""
1. Edit rules:
   ```bash
   sudo nano /etc/udev/rules.d/51-android.rules
   ```
2. Add:
   ```
   SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666", GROUP="plugdev"
   ```
3. Reload and restart:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   adb kill-server
   adb start-server
   ```
4. **Check device:**
   ```bash
   adb devices
   ```
   - It should now display as **device**.
"""



import logging
import re
import socket
import subprocess
import threading
import time
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import quaternion


# APK package name to check and ensure is running
APK_PACKAGE = "com.DefaultCompany.quest_reader_openxr"


def parse_message(msg: str) -> Dict[str, Any]:
    # Define regex patterns for parsing the data
    """Parse Quest3 data message."""
    patterns = {
        "Head": r"Head:\s+\(([^)]+)\)\s+\(Rotation:\s+\(([^)]+)\)",
        "Left Hand": r"Left Hand:\s+\(([^)]+)\)\s+\(Rotation:\s+\(([^)]+)\)",
        "Right Hand": r"Right Hand:\s+\(([^)]+)\)\s+\(Rotation:\s+\(([^)]+)\)",
        "Left Joystick": r"Left Joystick:\s+\(([^)]+)\)",
        "Right Joystick": r"Right Joystick:\s+\(([^)]+)\)",
        "Left Trigger": r"Left Trigger:\s+([\d.]+)",
        "Right Trigger": r"Right Trigger:\s+([\d.]+)",
        "Left Grip": r"Left Grip:\s+([\d.]+)",
        "Right Grip": r"Right Grip:\s+([\d.]+)",
        "A Button": r"A Button:\s+(True|False)",
        "B Button": r"B Button:\s+(True|False)",
        "X Button": r"X Button:\s+(True|False)",
        "Y Button": r"Y Button:\s+(True|False)",
    }

    parsed_data = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, msg)
        if match:
            if "Rotation" in key or "Hand" in key or "Head" in key:
                # Parse positions and rotations separately
                parsed_data[key] = {
                    "Position": tuple(map(float, match.group(1).split(", "))),
                    "Rotation": tuple(map(float, match.group(2).split(", "))),
                }
            elif "Joystick" in key:
                # Parse joystick coordinates
                parsed_data[key] = tuple(map(float, match.group(1).split(", ")))
            elif "Trigger" in key or "Grip" in key:
                # Parse single float values
                parsed_data[key] = float(match.group(1))
            elif "Button" in key:
                # Parse boolean values
                parsed_data[key] = match.group(1) == "True"

    return parsed_data


full_frame_pattern = re.compile(
    r"(Head: \(([-\d\.\, ]+)\) \(Rotation: \(([-\d\.\, ]+)\)\), "
    r"Left Hand: \(([-\d\.\, ]+)\) \(Rotation: \(([-\d\.\, ]+)\)\), "
    r"Right Hand: \(([-\d\.\, ]+)\) \(Rotation: \(([-\d\.\, ]+)\)\), "
    r"Left Joystick: \(([-\d\.\, ]+)\), "
    r"Right Joystick: \(([-\d\.\, ]+)\), "
    r"Left Trigger: ([-\d\.]+), "
    r"Right Trigger: ([-\d\.]+), "
    r"Left Grip: ([-\d\.]+), "
    r"Right Grip: ([-\d\.]+), "
    r"X Button: (True|False), "
    r"Y Button: (True|False), "
    r"A Button: (True|False), "
    r"B Button: (True|False))"
)


def process_data(raw_data: str) -> Optional[str]:
    """
    Try to find a full good frame.
    If found, return the full raw string of the latest frame.
    Else return None.
    """
    matches = full_frame_pattern.findall(raw_data)
    if matches:
        # matches[i][0] is the full matching chunk (because of how the regex is grouped)
        last_full_raw_chunk = matches[-1][0]
        return last_full_raw_chunk
    else:
        return None


class QuestReader:
    _instances: ClassVar[Dict[tuple, "QuestReader"]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()  # To ensure thread-safe initialization

    def __new__(cls, ip=None, port=12345, enable_low_pass_filter=False):
        # support multiple instances with different ip and port
        instance_key = (ip, port)

        if instance_key not in cls._instances:
            with cls._lock:
                if instance_key not in cls._instances:  # Double-checked locking
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, ip: Optional[str] = None, port: int = 12345, enable_low_pass_filter: bool = False):
        self.message_buffer = ""
        if self._initialized:
            return
        self.ip = ip
        self.port = port
        self.enable_low_pass_filter = enable_low_pass_filter

        # ADB mode when no IP provided
        if self.ip is None:
            self._setup_adb_mode()

        # Socket connection (works for both modes)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connect_ip = self.ip if self.ip else "127.0.0.1"
        self.client_socket.connect((connect_ip, self.port))
        logging.info(f"Connected to Unity server at {connect_ip}:{self.port}")

        self.lock = threading.Lock()
        self.data = None
        self.running = True  # Flag to control the thread
        self.start_reading()

        self.data_buffer = ""
        self.world_frame = np.eye(4)

        # Variables to check if the hand is stale
        self.left_hand_last_data = None
        self.left_hand_last_update_time = time.time()
        self.right_hand_last_data = None
        self.right_hand_last_update_time = time.time()
        self.hand_stale_time_threshold_s = 10  # 10 seconds

        while self.data is None:
            time.sleep(0.1)
            logging.info("Waiting for Quest3 data, Please check if the Quest3 is on and connected to the network...")

        self._initialized = True

    def _setup_adb_mode(self):
        """Setup ADB connection and ensure APK is running"""
        try:
            # Check ADB connection
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=False)
            if "device" not in result.stdout:
                raise Exception(
                    "No Quest3 device found via ADB. Make sure device is connected and ADB debugging is enabled."
                )

            logging.info("ADB connection established")

            # Clear all existing port forwarding first
            logging.info("Cleaning up any existing ADB port forwarding...")
            subprocess.run(["adb", "forward", "--remove-all"], capture_output=True, text=True, check=False)

            # Check if APK is installed
            cmd = ["adb", "shell", "pm", "list", "packages", APK_PACKAGE]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if APK_PACKAGE not in result.stdout:
                raise Exception(f"APK {APK_PACKAGE} is not installed on Quest3")

            # Check if APK is running
            cmd = ["adb", "shell", "ps | grep " + APK_PACKAGE]
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=False)

            if APK_PACKAGE not in result.stdout:
                logging.info(f"Starting APK {APK_PACKAGE}...")
                # Start the APK
                cmd = ["adb", "shell", "monkey", "-p", APK_PACKAGE, "-c", "android.intent.category.LAUNCHER", "1"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode != 0:
                    raise Exception(f"Failed to start APK {APK_PACKAGE}: {result.stderr}")
                time.sleep(3)  # Wait for app to start
                logging.info(f"APK {APK_PACKAGE} started successfully")
            else:
                logging.info(f"APK {APK_PACKAGE} is already running")

            # Setup port forwarding
            self._setup_port_forwarding()

        except Exception as e:
            logging.error(f"ADB setup failed: {e}")
            raise

    def _setup_port_forwarding(self):
        """Setup ADB port forwarding"""
        try:
            # Check if port forwarding already exists
            result = subprocess.run(["adb", "forward", "--list"], capture_output=True, text=True, check=False)
            expected_forward = f"tcp:{self.port}"

            if expected_forward in result.stdout:
                logging.info(f"ADB port forwarding already exists: localhost:{self.port} -> Quest3:{self.port}")
                return

            # Remove existing forwarding for this port
            subprocess.run(
                ["adb", "forward", "--remove", f"tcp:{self.port}"], capture_output=True, text=True, check=False
            )

            # Setup new port forwarding
            cmd = ["adb", "forward", f"tcp:{self.port}", f"tcp:{self.port}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                raise Exception(f"Failed to setup port forwarding: {result.stderr}")

            logging.info(f"ADB port forwarding setup: localhost:{self.port} -> Quest3:{self.port}")

        except Exception as e:
            logging.error(f"Port forwarding failed: {e}")
            raise

    def start_reading(self):
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.start()

    def _read_loop(self) -> None:
        while self.running:
            data = self._read()
            if data is None:
                continue
            with self.lock:
                self.data = data.copy()
            time.sleep(0.001)

    def _read(self) -> Optional[Dict[str, Any]]:
        data = self.client_socket.recv(4096)
        if data is None or data == b"":
            logging.error("No data received from Quest3, closing connection...")
            self.close()
            return None

        data = data.decode("utf-8")
        self.message_buffer += data
        # add extra check to prevent buffer from growing too large
        self.message_buffer = self.message_buffer[-5000:]
        # Find all head markers in the buffer
        head_markers = [i for i in range(len(self.message_buffer)) if self.message_buffer[i : i + 5] == "Head:"]

        # Try each potential frame from newest to oldest
        processed_frame = None
        for idx in reversed(range(len(head_markers))):
            start = head_markers[idx]
            end = len(self.message_buffer)
            if idx < len(head_markers) - 1:
                end = head_markers[idx + 1]

            # Extract the potential frame
            frame_data = self.message_buffer[start:end]
            # Clean up newlines
            if "\n" in frame_data:
                frame_data = frame_data.split("\n")[0]

            # Try to process this frame
            processed_frame = process_data(frame_data)
            if processed_frame is not None:
                # remove the processed frame from the buffer
                self.message_buffer = self.message_buffer[end:]
                break

        try:
            if processed_frame is None:
                logging.warning("QuestReader: No valid frame found in the message buffer")
            else:
                # Parse the string data into a dictionary
                parsed_data = parse_message(processed_frame)

                # Check if the hands are stale
                self._check_hand_stale(
                    "Left Hand",
                    parsed_data.get("Left Hand"),  # type: ignore
                    "left_hand_last_data",
                    "left_hand_last_update_time",
                )
                self._check_hand_stale(
                    "Right Hand",
                    parsed_data.get("Right Hand"),  # type: ignore
                    "right_hand_last_data",
                    "right_hand_last_update_time",
                )

                # Process the parsed data
                processed_data = {}
                for k, v in parsed_data.items():
                    key = k
                    if isinstance(v, dict):
                        try:
                            pos = np.array(v["Position"])
                            x, y, z, w = np.array(v["Rotation"])
                            quat = np.array([w, x, y, z])

                            rot_quat = quaternion.from_float_array(quat)
                            rot_mat = quaternion.as_rotation_matrix(rot_quat)
                            og_data = np.block([[rot_mat, pos.reshape(-1, 1)], [0, 0, 0, 1]])
                            if k == "Head":
                                og_head_data = og_data
                            # seems when hand sleeps, the position is not updated it will show  (-0.148, 0.000, -0.066)??? maybe check if y is 0
                            processed_data[key] = np.linalg.inv(self.world_frame) @ og_data
                        except Exception as e:
                            logging.debug(f"Error processing {k} data: {e}")
                            processed_data[key] = np.eye(4)
                    else:
                        processed_data[k] = v

                return processed_data
        except Exception as e:
            logging.debug(f"Parse error: {e}")
            return None

    def get_data(self) -> Dict[str, Any]:
        with self.lock:
            if self.data is None:
                return {}
            data = self.data.copy()

        return data

    def close(self) -> None:
        self.running = False  # Stop the thread
        if threading.current_thread() != self.read_thread:
            self.read_thread.join()  # Wait for the thread to finish if it's not the current thread
        self.client_socket.close()

        # Clean up ADB port forwarding if in ADB mode
        if self.ip is None:
            try:
                subprocess.run(
                    ["adb", "forward", "--remove", f"tcp:{self.port}"], capture_output=True, text=True, check=False
                )
                logging.info("ADB port forwarding cleaned up")
            except Exception as e:
                logging.debug(f"Failed to remove port forwarding: {e}")

        logging.info("Connection closed.")

    def _check_hand_stale(
        self,
        hand_name: str,
        hand_data: Dict[str, Any],
        last_data_attr: str,
        last_time_attr: str,
    ) -> None:
        if hand_data is not None:
            hand_tuple = (tuple(hand_data["Position"]), tuple(hand_data["Rotation"]))
            last_data = getattr(self, last_data_attr)
            last_time = getattr(self, last_time_attr)
            now = time.time()
            if last_data != hand_tuple:
                setattr(self, last_data_attr, hand_tuple)
                setattr(self, last_time_attr, now)
            elif now - last_time > self.hand_stale_time_threshold_s:
                logging.warning(
                    f"{hand_name} of Quest3 did not update for {self.hand_stale_time_threshold_s} seconds, it might be stale."
                )
                setattr(self, last_time_attr, now)  # prevent duplicate alarm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="IP address for network connection (if not provided, uses ADB)")
    parser.add_argument("--port", type=int, default=12345, help="Port for data connection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    reader = QuestReader(args.ip, args.port)

    while True:
        data = reader.get_data()
        print(f"data: {data}")
        time.sleep(0.5)
