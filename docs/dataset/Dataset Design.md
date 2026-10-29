# Dataset Design

Diseño del componente **Dataset** de Core, al nivel de detalle de `Learner Design.md`
y `Core Design.md`. Deriva de `../Architecture.md` (secciones "Core" y "Environment,
WorldModel, Reward Model y Dataset") y del `README.md` de esta carpeta.

Estado previo: la **interfaz** (`Dataset` / `AppendableDataset`, ABC) y el `FakeDataset`
de prueba ya existen desde la Fase 5 de Core y pasan la suite DSET. Lo que este documento
especifica es el paso que `CLAUDE.md` tenía como *fuera de alcance* y ahora se pide:
**adaptadores de datasets reales**, con un diseño que mantenga a Core agnóstico del
formato y de cómo se consulta cada corpus.

## Principio rector

**Core no conoce el formato ni la forma de consulta de ningún dataset.** Core sostiene
un objeto que cumple la ABC `Dataset` y solo invoca `sample(n, filters)` (y, cuando
corresponde, `add(pairs)`). Toda la lógica de "de dónde salen los datos y en qué formato
están" vive dentro de cada **adaptador**, que traduce su corpus al tipo canónico de
salida. Así se puede cambiar un corpus por otro —o sumar uno nuevo— sin tocar Core ni el
Learner.

```
Learner ──(protocolo)──> Core ──(ABC Dataset)──> Adaptador ──> corpus concreto
                           │                        (LeRobot, Open-X, local, …)
                     solo ve sample/add        traduce formato → tipo canónico
```

## La frontera: ABC `Dataset` / `AppendableDataset`

Único contrato que Core ve (ya implementado en `vla_core/dataset/`):

```python
class Dataset(ABC):
    @abstractmethod
    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        ...

class AppendableDataset(Dataset):
    @abstractmethod
    def add(self, pairs: list[tuple[Observation, ActionChunk]]) -> int:
        ...
```

- **Tipo canónico de salida.** Cada par es `(Observation, ActionChunk)` de `vla_common.types`:
  `Observation(images: dict[str, np.ndarray], proprio: np.ndarray, instruction: str)` y
  `ActionChunk = list[Action]` con `Action.values: np.ndarray` `float32`.
- **Unidades reales, sin normalizar.** El adaptador entrega las acciones tal como están
  en el corpus; cada política del Learner normaliza a su manera (ver `Learner Design.md`).
  Un adaptador **nunca** normaliza.
- **`filters`.** Diccionario opcional de criterios; cada adaptador respeta los que entiende
  e ignora el resto. El contrato exige `task` (filtra por instrucción); `embodiment` es
  opcional y solo tiene sentido en corpus multi-embodiment (Open-X).
- **Solo-lectura vs agregable.** Un corpus real es **de solo lectura**: implementa solo
  `Dataset`. `AppendableDataset` (el gancho de episodios exitosos, `Core.ingest_episode`)
  queda para datasets que aceptan reinyección (hoy `FakeDataset`; a futuro, un almacén
  agregable propio). Reinyectar en un corpus congelado no tiene sentido y rompería su
  reproducibilidad.

## Arquitectura pluggable

### Factory por `type`

Un `build_dataset(config) -> Dataset` (en `vla_core/dataset/factory.py`) despacha por el
campo `type` del bloque `dataset` de la config y construye el adaptador concreto. El
`builder.py` de Core lo invoca; el resto de Core no cambia.

```python
def build_dataset(config: dict) -> Dataset:
    tipo = config.get("type", "fake")
    if tipo == "fake":
        from vla_core.testing.fakes import FakeDataset
        return FakeDataset(...)
    if tipo == "local":
        from vla_core.dataset.local import LocalDataset
        return LocalDataset(path=config["path"], cameras=config.get("cameras"),
                            demo_chunk_size=config.get("demo_chunk_size", 1))
    if tipo == "lerobot":
        from vla_core.dataset.lerobot import LeRobotDataset   # grupo opcional
        return LeRobotDataset(repo_id=config["repo_id"], ...)
    if tipo == "openx":
        from vla_core.dataset.openx import OpenXDataset        # grupo opcional
        return OpenXDataset(...)
    raise ValueError(f"tipo de dataset desconocido: {tipo}")
```

- **Imports perezosos.** Cada adaptador se importa dentro de su rama, igual que hoy hace
  `builder._build_backend` con los fakes. Así un adaptador con dependencias pesadas no se
  importa salvo que la config lo pida, y `vla_core` base sigue sin `torch` ni `tensorflow`.
- **Dependencias en grupos opcionales uv.** `lerobot` y `openx` van en
  `[project.optional-dependencies]` (grupos `lerobot`, `openx`). La instalación base de Core
  no las trae. Se instalan con `uv sync --extra lerobot` cuando se quiere ese corpus.

### Adaptadores previstos

| `type`    | Clase            | Dependencias                | Grupo uv  | Append | Correr aquí |
|-----------|------------------|-----------------------------|-----------|--------|-------------|
| `fake`    | `FakeDataset`    | ninguna                     | base      | sí     | `fast`      |
| `local`   | `LocalDataset`   | ninguna (numpy)             | base      | no     | `fast`      |
| `lerobot` | `LeRobotDataset` | `lerobot` (HF, torch)       | `lerobot` | no     | `heavy`     |
| `openx`   | `OpenXDataset`   | `tensorflow-datasets`       | `openx`   | no     | `heavy`     |

- **`LocalDataset` (sin dependencias).** Lee un layout on-disk simple y autodescrito: un
  directorio de episodios, cada uno con sus arrays (`npz`: imágenes por cámara, `proprio`,
  `actions`) y su `instruction` en un `meta.json`. Es el **adaptador de referencia** y el
  soporte de los fixtures: al no depender de nada externo, su suite de contrato corre en
  `fast`, offline y determinista. Define de hecho el "formato canónico en disco" al que los
  adaptadores reales pueden exportar. En disco las imágenes pueden ir en `float32` (se
  entregan tal cual) o en `uint8` 0–255 (4× menos espacio; se entregan `float32` en
  `[0, 1]`). Carga todo en memoria al construirse, pero guarda **solo el frame de cada
  par**, convertido y copiado, no una vista del episodio: con LIBERO (273 mil frames a
  256×256, dos cámaras), retener los episodios completos en `float32` ocuparía ~430 GB.
  Los corpus grandes se convierten con `scripts/prepare_lerobot_local.py` (ver
  `Pi0 E2E Plan.md`).
- **`LeRobotDataset` (primer corpus real).** Traduce un dataset LeRobot (HuggingFace,
  parquet + vídeos) al tipo canónico. Liviano (apoyado en `torch`/HF datasets), cabe en la
  RTX 2050 y funciona offline contra un snapshot local. Primer adaptador real a construir.
- **`OpenXDataset` (después).** Open-X Embodiment en formato RLDS vía `tensorflow-datasets`.
  Multi-embodiment (de ahí el filtro `embodiment`). Dependencia pesada (`tensorflow`); se
  deja como adaptador posterior, posiblemente stub hasta que se pida.

## Mapeo corpus → tipo canónico

Responsabilidad de cada adaptador, documentada por adaptador en su docstring:

1. **Imágenes.** Cada cámara del corpus → entrada de `Observation.images` (`float32`). La
   config `cameras` declara qué claves tomar y con qué nombre exponerlas; el orden de ejes
   se normaliza al que usa el harness (ver `vla_common`).
2. **Propriocepción.** Vector de estado del corpus → `Observation.proprio` (`float32`).
3. **Instrucción.** Lenguaje natural de la tarea → `Observation.instruction`. Es también lo
   que filtra `filters["task"]`.
4. **Acciones.** Secuencia de acciones del corpus → `ActionChunk`, en **unidades reales**,
   `float32`, sin normalizar.
5. **Chunking.** Se agrupan `demo_chunk_size` pasos consecutivos por `ActionChunk`. Si un
   episodio termina a mitad de chunk, el chunk se corta en ese punto (misma regla de
   episodio que Core: un chunk = una transición).

## Configuración

Bloque `dataset` dentro de la config de Core. Ejemplos:

```yaml
# configs/core/fake.yaml  (ya existe, vía backend fake)
dataset: { type: fake, demo_chunk_size: 1 }

# configs/core/local.yaml
dataset:
  type: local
  path: data/demos/reach_local       # directorio de episodios
  cameras: { front: image }          # clave canónica -> clave del corpus
  demo_chunk_size: 1

# configs/core/lerobot.yaml
dataset:
  type: lerobot
  repo_id: lerobot/<dataset>         # o ruta a snapshot local
  cameras: { front: observation.images.top }
  demo_chunk_size: 4
```

## Tests

- Todo adaptador **hereda `DatasetContract`** (`testing/contracts/dataset_contract.py`,
  DSET01–03). Un adaptador de solo lectura hereda solo `DatasetContract`; uno agregable,
  `AppendableDatasetContract` (DSET04–05).
- **`LocalDataset` corre el contrato en `fast`** contra un **fixture chico** incluido en el
  repo (formato `local`, determinista, offline). El fixture es el mismo mecanismo con que
  se prueban adaptadores reales sin bajar nada.
- **Adaptadores reales (`lerobot`, `openx`) → tests marcados `heavy`**: se escriben pero no
  se corren en esta máquina (requieren el corpus completo). Nada de descargas en `fast`.
- Un corpus descargado es **dato no confiable**: se guarda en su propio directorio, se
  accede por ruta y nunca se ejecuta código desde ahí.

## Decisiones

- **D1 — Core agnóstico del formato.** La ABC `Dataset` es la única frontera; toda la lógica
  de formato/consulta vive en adaptadores. *Motivo:* pedido explícito ("Core que se abstraiga
  de cómo se consulta o el formato"); además es la regla de `Architecture.md` ("cambiar un
  dataset por otro sin tocar el Learner").
- **D2 — Factory `build_dataset(config)` por `type`, imports perezosos.** *Motivo:* mismo
  patrón que `builder._build_backend`; mantiene `vla_core` base sin `torch`/`tensorflow`.
- **D3 — Dependencias reales en grupos opcionales uv (`lerobot`, `openx`).** *Motivo:* regla
  de `CLAUDE.md` (base liviana; pesadas en grupos opcionales); no se agrega ninguna dep sin
  pedirlo.
- **D4 — Adaptadores reales de solo lectura (`Dataset`, no `AppendableDataset`).** *Motivo:*
  reinyectar en un corpus congelado rompe su reproducibilidad; el gancho de episodios queda
  en datasets agregables.
- **D5 — `LocalDataset` sin dependencias como adaptador de referencia y base de fixtures.**
  *Motivo:* permite correr el contrato de un adaptador no-fake en `fast`, offline y
  determinista, y fija el formato canónico en disco.
- **D6 — Filtros: `task` obligatorio en el contrato; `embodiment` opcional.** *Motivo:*
  `embodiment` solo aplica a corpus multi-embodiment (Open-X).
- **D7 — `vla_core/dataset.py` pasa a paquete `vla_core/dataset/`.** `__init__.py` reexporta
  `Dataset`/`AppendableDataset` (la ruta de import `vla_core.dataset` no cambia) y aloja
  `factory.py`, `local.py`, `lerobot.py`, `openx.py`. *Motivo:* alinea con el diagrama de
  estructura de `CLAUDE.md` (que ya muestra `dataset/`) sin romper imports existentes.
- **D8 — `LeRobot` como primer corpus real.** *Motivo:* liviano, cabe en la RTX 2050, offline
  con snapshot local; Open-X (tensorflow) queda para después.

## Plan de acción

Cada fase cierra con `uv run pytest -m fast`, `ruff` y `pyright` en verde, y un commit en la
rama `Dataset`.

1. ✅ **Refactor a paquete + factory.** `dataset.py` → `dataset/` (`base.py` con las ABCs,
   reexportadas desde `__init__.py`) + `dataset/factory.py` con `build_dataset`. Sin cambios
   de comportamiento: `type: fake` sigue igual. Suite `test_dataset_factory.py` (`fast`).
2. ✅ **`LocalDataset` + fixture + contrato.** `dataset/local.py` (sample, filtros, mapeo de
   cámaras, chunking), fixture chico en `tests/data/local_reach/` y `test_dataset_local.py`
   corriendo `DatasetContract` en `fast`. `build_core` inyecta el `dataset` del YAML en el
   Core fake (`make_fake_core(dataset=...)`); config `configs/core/local.yaml` validada por
   `test_configs.py`.
3. ✅ **`LeRobotDataset` (grupo opcional).** Grupo opcional uv `lerobot` en
   `packages/vla_core/pyproject.toml` (`uv.lock` regenerado), `dataset/lerobot.py` con el
   mapeo documentado (import perezoso; no entra al venv base), `type: lerobot` en la factory,
   config `configs/core/lerobot.yaml` y `test_dataset_lerobot.py` marcada `heavy` (gated por
   `VLA_LEROBOT_REPO_ID`, no corre aquí). En `fast`, `test_dataset_factory.py` verifica que
   construir `lerobot` sin el grupo da un `ImportError` claro.
4. ✅ **`OpenXDataset` (grupo opcional).** Grupo opcional uv `openx`
   (`tensorflow-datasets`), `dataset/openx.py` (RLDS/TFDS, filtro `embodiment`, import
   perezoso), `type: openx` en la factory, config `configs/core/openx.yaml` y
   `test_dataset_openx.py` marcada `heavy` (gated por `VLA_OPENX_NAME`/`VLA_OPENX_BUILDER_DIR`).
   En `fast`, `test_dataset_factory.py` verifica el `ImportError` claro sin el grupo.
5. ✅ **Docs + alcance.** Actualizados `docs/dataset/README.md`, la sección "Datos" y el
   diagrama de estructura de `Core Design.md` (`dataset.py` → `dataset/`) y el "Alcance" de
   `CLAUDE.md` (`local` y `lerobot` implementados; solo `openx` queda fuera).

## Desviaciones respecto al diseño previo

- `CLAUDE.md` listaba "el Dataset real (Open-X / LeRobot)" como **fuera de alcance**. Este
  documento lo incorpora como adaptadores opcionales; se actualizará el "Alcance" de
  `CLAUDE.md` en la Fase 5 del plan, al implementarse.
- `Core Design.md` muestra la estructura con `dataset.py`; se reconcilia con el paquete
  `dataset/` (D7). El diagrama de estructura del repo en `CLAUDE.md` ya usa `dataset/`.
