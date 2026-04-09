import type { RobotPoseFrame } from "./robotPose";

type PollerOptions = {
  intervalMs: number;
  onPose: (frame: RobotPoseFrame) => void;
  onDisconnect?: () => void;
};

export function createPosePoller(url: string, options: PollerOptions) {
  let stopped = false;

  const tick = async () => {
    if (stopped) {
      return;
    }
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Unexpected status ${response.status}`);
      }
      const frame = (await response.json()) as RobotPoseFrame;
      options.onPose(frame);
    } catch {
      options.onDisconnect?.();
    } finally {
      if (!stopped) {
        window.setTimeout(tick, options.intervalMs);
      }
    }
  };

  window.setTimeout(tick, options.intervalMs);

  return () => {
    stopped = true;
  };
}
