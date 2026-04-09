from __future__ import annotations

import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_mock_pose_frame(time_seconds: float) -> dict:
    shoulder = 0.35 * math.sin(time_seconds * 2.2)
    elbow = 0.55 * math.sin(time_seconds * 2.2 + math.pi / 2.0)
    return {
        "root_position": [0.0, 0.0, 0.0],
        "root_quaternion": [0.0, 0.0, 0.0, 1.0],
        "joints": {
            "front_left_shoulder": shoulder,
            "front_left_elbow": elbow,
            "front_right_shoulder": -shoulder,
            "front_right_elbow": -elbow,
            "back_left_shoulder": -shoulder,
            "back_left_elbow": -elbow,
            "back_right_shoulder": shoulder,
            "back_right_elbow": elbow,
        },
    }


class PoseHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        if self.path == "/pose":
            self._send_json(build_mock_pose_frame(time.time()))
            return
        self._send_json({"error": "not found"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), PoseHandler)
    print("pose bridge listening on http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
