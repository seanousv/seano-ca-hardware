#!/usr/bin/env bash
set +e

echo "[INFO] sending graceful SIGINT to phase7 launch"
pkill -INT -f "phase7_cuav_usb_hardware.launch.py" || true

sleep 8

echo "[INFO] cleaning leftover ROS processes if any"
pkill -TERM -f "mavros_node" || true
pkill -TERM -f "camera_node" || true
pkill -TERM -f "detector_node" || true
pkill -TERM -f "risk_evaluator_node" || true
pkill -TERM -f "mission_mode_manager_node" || true
pkill -TERM -f "watchdog_failsafe_node" || true
pkill -TERM -f "command_mux_node" || true
pkill -TERM -f "actuator_safety_limiter_node" || true
pkill -TERM -f "mavros_rc_override_bridge_node" || true
pkill -TERM -f "auto_controller_stub_node" || true
pkill -TERM -f "event_logger_node" || true
pkill -TERM -f "web_video_server" || true

sleep 2

echo "[INFO] force-killing stubborn leftovers only"
pkill -KILL -f "phase7_cuav_usb_hardware.launch.py" || true
pkill -KILL -f "mavros_node" || true
pkill -KILL -f "camera_node" || true
pkill -KILL -f "detector_node" || true
pkill -KILL -f "risk_evaluator_node" || true
pkill -KILL -f "mission_mode_manager_node" || true
pkill -KILL -f "watchdog_failsafe_node" || true
pkill -KILL -f "command_mux_node" || true
pkill -KILL -f "actuator_safety_limiter_node" || true
pkill -KILL -f "mavros_rc_override_bridge_node" || true
pkill -KILL -f "auto_controller_stub_node" || true
pkill -KILL -f "event_logger_node" || true
pkill -KILL -f "web_video_server" || true

echo "[OK] phase7 stopped"
