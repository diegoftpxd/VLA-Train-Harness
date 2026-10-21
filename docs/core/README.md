# Core

Gateway entre el Learner y el resto del sistema: enruta pedidos, pide recompensas, sirve datos supervisados y registra cada episodio. Proceso CPU, sin modelo. **Fase actual** (`packages/vla_core`). Responsabilidades, interfaz hacia el Learner, interfaces que consume y protocolo ZeroMQ en [`../Architecture.md`](../Architecture.md) (sección "Core").

Documentos:

- [`Core Design.md`](./Core%20Design.md) — clases e interfaces internas de Core, al nivel de detalle de `../learner/Learner Design.md`. Especificación a implementar.
- [`Test Core.md`](./Test%20Core.md) — suites de contrato de Core y de sus componentes.
