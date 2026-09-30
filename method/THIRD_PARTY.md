# Upstream components and adaptations

This directory is our assembled implementation, NOT a claim that all algorithms are original.
CaP-X: capgym/cap-x, revision 53e9966d7a8e2fa7494676772bccc35280f5c0ed.
Original task prompts, simulator, PyRoKi motion API, controller waits and wipe raster trajectory pattern remain attributable to that project.
https://github.com/capgym/cap-x

Laya: the public implementation under /root/autodl-tmp/laya_wm_pilot_20260929/vendor/laya and public base weights; T4 is this project's separately trained semantic adaptation.
GroundingDINO: public pretrained detection model, reused rather than trained here.
MuJoCo, robosuite, PyRoKi, Transformers, PyTorch and their dependencies retain their own licenses.
Exact public source/revision references are retained in the existing server model/runtime metadata.
Selected motion method bodies were extracted unchanged; import isolation and RGBD/DINO geometry are adaptations.
Our question compiler, graph integration and recovery assembly are separate from the pretrained competence and upstream skills.
Do not attribute public raster trajectories, low-level IK or paper benchmark results to our method.
Licenses present in source locations are copied below; missing files are listed, not guessed.

[
  {
    "component": "capx",
    "source": "/root/capx_official_runtime_20260930/cap-x-53e9966d7a8e2fa7494676772bccc35280f5c0ed",
    "copied": [
      "LICENSE"
    ]
  },
  {
    "component": "laya",
    "source": "/root/autodl-tmp/laya_wm_pilot_20260929/vendor/laya",
    "copied": [
      "LICENSE"
    ]
  },
  {
    "component": "grounding_dino",
    "source": "/root/autodl-tmp/laya_wm_pilot_20260929/models/grounding_dino_tiny",
    "copied": []
  }
]
