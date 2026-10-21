# Environment

Ambiente real o simulado, con API tipo Gymnasium (`reset`, `step`, `close`). Un proceso por ambiente o por grupo vectorizado. Declara su espacio de acciones y sus cámaras al registrarse en Core. Ver [`../Architecture.md`](../Architecture.md) (sección "Environment, WorldModel, Reward Model y Dataset").

**Fuera de alcance**: el Environment real no se implementa. Core se construye y se prueba contra un Environment *fake* en CPU. La documentación de diseño de los fakes vive con el código de Core (`packages/vla_core/.../testing/`).
