"""Оценка обученной модели без exclusions."""
import sys
from pathlib import Path

sys.path.insert(0, '/Users/kira/Downloads')
import experiment  

import hydra
from omegaconf import DictConfig, OmegaConf
from grappa.training import Experiment


@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="train_no_excl")
def main(cfg: DictConfig) -> None:
    print(f"ref_terms: {cfg.data.data_module.ref_terms}")
    print(f"datasets: {cfg.data.data_module.datasets}")

    import glob
    run_dirs = []
    for pattern in ['grappa-no-excl-spice/*/', 'grappa-no-excl-3ds/*/', 'grappa-no-excl/*/']:
        run_dirs.extend(Path('ckpt/grappa-experiment').glob(pattern))
    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No checkpoint directories found")
    run_dir = run_dirs[0]

    ckpts = list(run_dir.glob('epoch:*.ckpt'))
    if not ckpts:
        ckpts = [run_dir / 'last.ckpt']
    ckpt_path = ckpts[0]
    print(f'Using: {ckpt_path}')
    cfg.experiment.ckpt_path = str(ckpt_path)
    cfg.experiment.use_wandb = False

    experiment = Experiment(config=cfg, is_train=False, load_data=True)
    experiment.test(
        ckpt_path=ckpt_path,
        n_bootstrap=10,
        load_split=False,
        store_test_data=True,
    )


if __name__ == "__main__":
    main()
