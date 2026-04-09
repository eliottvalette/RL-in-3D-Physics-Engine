import { DEFAULT_RIG_MAP, type RobotRigMap } from "@/lib/robotRigMap";

type Props = {
  rigMap?: RobotRigMap;
};

export function RobotRigDebug({ rigMap = DEFAULT_RIG_MAP }: Props) {
  return (
    <div>
      <h2>Rig map</h2>
      <p>
        This is the explicit logical mapping between the RL joints and the GLB bones. The axis and
        sign values are the first thing to validate visually.
      </p>

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
