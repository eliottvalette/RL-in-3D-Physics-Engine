# three_rendering

Mini-workspace Next.js pour visualiser le robot GLB avec un mapping de joints explicite, puis le piloter plus tard depuis le runtime Python du quadruped.

## Scope

- charger `public/models/rigged_robot_walk.glb`
- exposer un viewer Three.js local
- figer un `rig_map` logique entre joints RL et bones GLB
- fournir un bridge Python local minimal pour servir des poses

## Structure

```text
three_rendering/
  app/
    components/
      RobotRigDebug.tsx
      RobotViewer.tsx
    globals.css
    layout.tsx
    page.tsx
  data/
    pose_frame.example.json
    rig_map.example.json
  lib/
    robotModel.ts
    robotPose.ts
    robotRigMap.ts
    socketClient.ts
  public/
    models/
      rigged_robot_walk.glb
  scripts/
    extract_rig_map.ts
    inspect_glb.ts
  server/
    pose_bridge.py
```

## Commands

```bash
npm run dev
npm run build
npm run inspect-glb
npm run extract-rig-map
python server/pose_bridge.py
```

## Contract

Le viewer ne joue pas le clip d'animation. Il applique une pose explicite sur les bones utilises par le quadruped RL :

- `front_left_shoulder`
- `front_left_elbow`
- `front_right_shoulder`
- `front_right_elbow`
- `back_left_shoulder`
- `back_left_elbow`
- `back_right_shoulder`
- `back_right_elbow`

Le bridge Python actuel sert une pose mockable via HTTP. Le branchement direct sur le `.pth` et sur l'env Python sera l'etape suivante.
