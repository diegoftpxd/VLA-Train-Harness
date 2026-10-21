# Dataset

Demos supervisadas en el formato de `Transition` (`source="dataset"`), con acciones en unidades reales sin normalizar. En la v1 **no es un proceso propio: vive dentro de Core** y solo se accede a través de él (`sample_supervised`). Los episodios exitosos que registra Core pueden agregarse aquí tras un filtro. Ver [`../Architecture.md`](../Architecture.md) (secciones "Core" y "Environment, WorldModel, Reward Model y Dataset").

El diseño detallado (frontera ABC, factory `build_dataset`, adaptadores, mapeo canónico, decisiones y plan) está en [`Dataset Design.md`](Dataset%20Design.md). Core es agnóstico del formato: solo ve la ABC `Dataset` y la factory elige el adaptador por `type`.

**Adaptadores**: `fake` (en memoria), `local` (layout on-disk, sin dependencias, corre en `fast`) y `lerobot` (corpus LeRobot, grupo opcional uv `lerobot`, suite `heavy`). **Fuera de alcance** hasta que se pida: `openx` (Open-X Embodiment, tensorflow).
