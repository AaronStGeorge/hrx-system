# Model publication

The authoritative workflow is [Sharing models on Hugging Face](https://rocm.github.io/hrx-system/loom/serving/sharing/hugging-face/).
[Model documents](https://rocm.github.io/hrx-system/loom/serving/model-documents/)
and [model programs](https://rocm.github.io/hrx-system/loom/serving/model-programs/)
describe the contracts. [Kernel authoring](https://rocm.github.io/hrx-system/loom/workflows/agent-driven-kernel-development/)
covers correctness and performance evidence.

`model/*/model.json` files are authored. `python refresh.py` regenerates their
asset locks and `catalog.json`; `python smoke.py` verifies without refreshing.
Script options live in `--help`. The README describes this model's numerical
contract. A replacement model needs its own expected-output check.
