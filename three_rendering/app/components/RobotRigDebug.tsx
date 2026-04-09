import { DEFAULT_RIG_MAP, type RobotRigMap } from "@/lib/robotRigMap";
import {
  DEFAULT_BOX_QUADRUPED_CALIBRATION,
  type BoxQuadrupedCalibration,
} from "@/lib/boxQuadrupedConfig";

type Props = {
  rigMap?: RobotRigMap;
  calibration?: BoxQuadrupedCalibration;
};

export function RobotRigDebug({
  rigMap = DEFAULT_RIG_MAP,
  calibration = DEFAULT_BOX_QUADRUPED_CALIBRATION,
}: Props) {
  return (
    <div>
      <h2>Rig map</h2>
      <p>
        The GLB rig and the 9-box procedural quadruped are rendered together and driven by the same
        pose frame. Adjust the calibration first, then port it to Python.
      </p>

      <h3>Box calibration</h3>
      <div className="debug-list">
        <div className="debug-item">
          <strong>bodySize</strong>
          <code>{JSON.stringify(calibration.bodySize)}</code>
        </div>
        <div className="debug-item">
          <strong>upperLegSize</strong>
          <code>{JSON.stringify(calibration.upperLegSize)}</code>
        </div>
        <div className="debug-item">
          <strong>lowerLegSize</strong>
          <code>{JSON.stringify(calibration.lowerLegSize)}</code>
        </div>
        <div className="debug-item">
          <strong>bodyCenterOffset</strong>
          <code>{JSON.stringify(calibration.bodyCenterOffset)}</code>
        </div>
      </div>

      <h3 style={{ marginTop: 16 }}>RL to GLB mapping</h3>

      <div className="debug-list">
        {Object.entries(rigMap.joints).map(([jointName, mapping]) => (
          <div className="debug-item" key={jointName}>
            <strong>{jointName}</strong>
            <div>
              bone: <code>{mapping.boneName}</code>
            </div>
            <div>
              axis/sign: <code>{mapping.axis}</code> / <code>{mapping.sign}</code>
            </div>
            <div>
              rest: <code>{mapping.restAngle}</code>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
