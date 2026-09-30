# Third-party hand assets

The repository's [MIT license](LICENSE) covers this project's own code. Hand models and assets obtained from other projects retain their respective upstream terms. Keep these notices with any redistribution of the corresponding assets.

## Allegro Hand

`source/dynamic_dexgrasp_lab/assets/hands/allegro_hand/allegro.py` imports `ALLEGRO_HAND_CFG` from [Isaac Lab's Allegro configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/allegro.py). Isaac Lab's source code is [BSD-3-Clause](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE). This repository does not include an Allegro mesh or USD model: the runtime asset comes from the installed Isaac Lab / Isaac Sim stack. Isaac Lab's code license should not be taken as a separate license grant for any externally hosted robot model it loads.

## Inspire Hand

The original left- and right-hand URDFs under `source/dynamic_dexgrasp_lab/assets/hands/inspire_hand/xml/` match the XML content of [DexSuite dex-urdf's Inspire Hand models](https://github.com/dexsuite/dex-urdf/tree/main/robots/hands/inspire_hand). DexSuite's [Robot Source table](https://github.com/dexsuite/dex-urdf#robot-source) credits **Inspire-Robot** and lists the Inspire Hand source asset license as [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The Inspire URDFs, meshes, converted MJCF/XML files, and derived USD assets in this repository should be treated as subject to those source-asset terms. Credit Inspire-Robot and DexSuite, link the license, indicate modifications, use only in permitted noncommercial contexts, and apply the required share-alike terms to adapted material. The [MIT license of the dex-urdf repository](https://github.com/dexsuite/dex-urdf/blob/main/LICENSE) does not replace the Inspire model's separately listed source-asset license.

This repository reformatted the source URDFs and generated simulation-oriented MJCF/XML, collision meshes, and USD variants from the hand model. This notice documents the known provenance and upstream license statement; it does not grant additional rights from the original rights holder.
