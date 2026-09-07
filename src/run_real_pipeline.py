from __future__ import annotations

import sys
from pathlib import Path

from rag_pipeline.config import ExperimentConfig
from rag_pipeline.pipeline import RealPipelineRunner


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/real_pipeline.example.json")
    config = ExperimentConfig.from_json(config_path)
    runner = RealPipelineRunner(config)
    artifacts = runner.run()
    runner.write_outputs(artifacts)
    print(f"Wrote outputs to {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
