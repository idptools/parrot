"""
Backwards-compatible names for the original PARROT networks.

Code written against earlier versions of PARROT (e.g. SPARROW) builds networks
with `BRNN_MtO(input_size, hidden_size, num_layers, num_classes, device)` or
`BRNN_MtM(...)` and loads saved weights into them. Both are now thin wrappers
around parrot.architectures.brnn.BRNN, with identical parameter names, so that
code keeps working. New code should use parrot.architectures.build_network().

.............................................................................
idptools-parrot was developed by the Holehouse lab
     Original release ---- 2020

Question/comments/concerns? Raise an issue on github:
https://github.com/idptools/parrot

Licensed under the MIT license.
"""

from parrot.architectures.brnn import BRNN, run_lstm  # noqa: F401  (run_lstm re-exported for older imports)
from parrot.tasks import Task


class BRNN_MtM(BRNN):
    """Many-to-many BRNN: one output per residue (residue-mapped tasks).

    Parameters
    ----------
    input_size : int
        Length of the input vectors at each timestep
    hidden_size : int
        Size of hidden vectors in the network
    num_layers : int
        Number of hidden layers (for each direction) in the network
    num_classes : int
        Number of classes, or 1 for a regression problem
    device : str
        Kept for backwards compatibility; move the network with .to(device)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, device: str) -> None:
        """Build a residue-mapped BRNN (parameters as described for the class)."""
        super().__init__(input_size, Task('residues', num_classes), hidden_size=hidden_size, num_layers=num_layers,
                         attention_pooling=False, loglength_feedforward=False)
        self.device = device


class BRNN_MtO(BRNN):
    """Many-to-one BRNN: one output per sequence (sequence-mapped tasks).

    Parameters
    ----------
    input_size : int
        Length of the input vectors at each timestep
    hidden_size : int
        Size of hidden vectors in the network
    num_layers : int
        Number of hidden layers (for each direction) in the network
    num_classes : int
        Number of classes, or 1 for a regression problem
    device : str
        Kept for backwards compatibility; move the network with .to(device)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, device: str) -> None:
        """Build a sequence-mapped BRNN (parameters as described for the class)."""
        super().__init__(input_size, Task('sequence', num_classes), hidden_size=hidden_size, num_layers=num_layers,
                         attention_pooling=False, loglength_feedforward=False)
        self.device = device
