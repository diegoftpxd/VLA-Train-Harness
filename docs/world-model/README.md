# WorldModel

Predice cómo cambian las observaciones ante una secuencia de acciones. Trabaja en su espacio latente: `encode` convierte una observación real en estado latente y `rollout` simula varias secuencias desde ahí. Se usa preentrenado y congelado. Ver [`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model y Dataset").

**Fuera de alcance**: el WorldModel real no se implementa (ni se entrena: eso es fuera de este harness). Core se construye y se prueba contra un WorldModel *fake* en CPU.
