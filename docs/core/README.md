# Core

Gateway entre el Learner y el resto del sistema: enruta pedidos, pide recompensas, sirve datos supervisados y registra cada episodio. Proceso CPU, sin modelo. **Terminado** (`packages/vla_core`). Responsabilidades, interfaz hacia el Learner, interfaces que consume y protocolo ZeroMQ en [`../Architecture.md`](../Architecture.md) (sección "Core").

Documentos:

- [`Core Reference.md`](./Core%20Reference.md) — **manual de uso** del Core implementado: API, cómo levantarlo, componentes, flujos, fallas, configuración, módulos y extensión. Empieza por acá para usar o entender Core.
- [`Core Design.md`](./Core%20Design.md) — clases e interfaces internas y la racionalidad del diseño, al nivel de detalle de `../learner/Learner Design.md`.
- [`Test Core.md`](./Test%20Core.md) — suites de contrato de Core y de sus componentes.
