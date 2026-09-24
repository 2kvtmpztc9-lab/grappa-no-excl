# GraPPA без exclusions — рабочая модель

## Что это

ML-FF на базе GraPPA, которая:
- **сама предсказывает** `q, σ, ε` (nonbonded) для каждого атома,
- обучается **напрямую на QM** (`E_ref = E_QM`, без Δ-learning),
- считает nonbonded **по всем парам** с damping (без exclusions при обучении),
- **MD стабильна** с PME + CustomNonbondedForce.

## Модель

**Датасет:** `spice-dipeptide` (677 молекул, SPICE QM-уровень).
**Эпох:** 309 (early stopping).
**Чекпоинт:** `checkpoint.ckpt`.

## Метрики на test set

- `rmse_energies = 2.676` ккал/моль (`ratio_E = 0.128` — модель объясняет **98.4%** разброса QM)
- `rmse_gradients = 5.549` ккал/моль/Å (`ratio_G = 0.258`)
- n_confs = 2687

## Параметры (средние)

- `q`: mean = 0.0006, std = 0.342
- `σ`: mean = 2.675 Å, std = 0.673
- `ε`: mean = 0.146 ккал/моль, std = 0.161

## MD (убиквитин, 1UBQ)

- **Система:** `1ubq_system_pme.xml` — 1231 атом, PME для Coulomb + CustomNonbondedForce для LJ + damping, exceptions 1-2, 1-3.
- **Минимизация:** -7450 ккал/моль.
- **10 ps MD:** -8436 ккал/моль — **стабильно**.

## Как использовать

```python
from grappa.utils.model_loading_utils import model_from_path
model = model_from_path('checkpoint.ckpt')
model.eval()
