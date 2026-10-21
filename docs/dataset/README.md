# Dataset

Demos supervisadas en el formato de `Transition` (`source="dataset"`), con acciones en unidades reales sin normalizar. En la v1 **no es un proceso propio: vive dentro de Core** y solo se accede a través de él (`sample_supervised`). Los episodios exitosos que registra Core pueden agregarse aquí tras un filtro. Ver [`../Architecture.md`](../Architecture.md) (secciones "Core" y "Environment, WorldModel, Reward Model y Dataset").

**Fuera de alcance**: el Dataset real (un corpus concreto, p. ej. Open-X / LeRobot) no se implementa; Core usa un Dataset *fake*. El adaptador real iría en un grupo opcional de uv cuando se pida.
