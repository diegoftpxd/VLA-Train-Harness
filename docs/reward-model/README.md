# Reward Model

Proceso propio; Core le habla solo por `score`. Recibe la instrucción y la transición (observación, acción, observación siguiente) y devuelve una recompensa. Puede ser un detector de éxito del simulador, un modelo aprendido o un VLM como juez. Acepta batches. Ver [`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model y Dataset").

**Fuera de alcance**: el Reward Model real no se implementa. Core se construye y se prueba contra un Reward Model *fake* en CPU.
