import { RobotRigDebug } from "./components/RobotRigDebug";
import { RobotViewer } from "./components/RobotViewer";
import { DEFAULT_RIG_MAP } from "@/lib/robotRigMap";

export default function Home() {
  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">three_rendering</p>
          <h1>Robot GLB viewer and rig mapping workspace</h1>
          <p className="lede">
            The GLB is rendered directly. Joint angles are applied on explicit bones instead of
            replaying the baked walk clip.
          </p>
        </div>
      </section>

      <section className="viewer-grid">
        <div className="viewer-panel">
          <RobotViewer />
        </div>
        <aside className="debug-panel">
          <RobotRigDebug rigMap={DEFAULT_RIG_MAP} />
        </aside>
      </section>
    </main>
  );
}
