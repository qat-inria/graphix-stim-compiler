"""Compilation pass for Clifford maps using stim functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import stim
from graphix.circ_ext.extraction import CliffordMap, PauliString
from graphix.fundamentals import Axis, Sign

if TYPE_CHECKING:
    from graphix.transpiler import Circuit

    _SYNTH_METHOD: TypeAlias = Literal["elimination", "graph_state"]


def cm_stim_pass(clifford_map: CliffordMap, circuit: Circuit) -> None:
    """Add a Clifford map to a circuit by using stim's tableau synthesis.

    The input circuit is modified in-place. This function assumes that the Clifford Map has been remap, i.e., its Pauli strings are defined on qubit indices instead of output nodes. See :meth:`PauliString.remap` for additional information.

    Parameters
    ----------
    clifford_map: CliffordMap
        The Clifford map to be transpiled. Its Pauli strings are assumed to be defined on qubit indices.
    circuit : Circuit
        The circuit to which the operation is added. The input circuit is assumed to be compatible with ``CliffordMap.input_nodes`` and ``CliffordMap.output_nodes``.

    Notes
    -----
    This pass only handles unitaries (Clifford maps with the same number of input and ouptut nodes).

    Gate set: H, S, CNOT
    """

    def to_stim_circuit(clifford_map: CliffordMap, method: _SYNTH_METHOD) -> stim.Circuit:
        """Transpile the Clifford map into a stim circuit.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be transpiled.
        method: Literal["elimination", "graph_state"]
            Stim method for synthetizing the circuit.

        Returns
        -------
        stim.Circuit

        Notes
        -----
        See https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Tableau.to_circuit for additional information.
        """
        return cm_to_stim_tableau(clifford_map).to_circuit(method)

    stim_circuit = to_stim_circuit(clifford_map, method="elimination")  # Gate set: H, S, CX

    # "stim.Circuit" has no attribute "__iter__"
    # (but __len__ and __getitem__)
    instruction: stim.CircuitInstruction
    for instruction in stim_circuit:  # type: ignore[attr-defined]
        match instruction.name:
            case "CX":
                for control, target in instruction.target_groups():
                    assert control.qubit_value is not None
                    assert target.qubit_value is not None
                    circuit.cnot(control.qubit_value, target.qubit_value)
            case "H":
                for (qubit,) in instruction.target_groups():
                    assert qubit.qubit_value is not None
                    circuit.h(qubit.qubit_value)
            case "S":
                for (qubit,) in instruction.target_groups():
                    assert qubit.qubit_value is not None
                    circuit.s(qubit.qubit_value)


def pauli_string_to_stim(ps: PauliString, n_qubits: int) -> stim.PauliString:
    """Transform a :class:`graphix.circ_ext.extraction.PauliString` into a :class:`stim.PauliString` instance.

    This function assumes that the Pauli string has been remap, i.e., it is defined on qubit indices and not on node values. See :meth:`graphix.circ_ext.PauliString.remap` for additional information.

    Parameters
    ----------
    ps: PauliString
        The Pauli string to be transformed.
    n_qubits : int
        Width of the circuit on which the Pauli string is defined.

    Returns
    -------
    stim.PauliString
        The Pauli string in `stim` format.

    Raises
    ------
    ValueError
        If the Pauli string is not compatible with ``n_qubits``.

    Notes
    -----
    Qubits not appearing in ``ps.axes.keys`` are assigned the identity operator in the returned `stim.PauliString`.
    """
    if not all(0 <= node < n_qubits for node in ps.axes):
        raise ValueError("The Pauli string contains qubit indices beyond the circuit's width.")

    pauli_str = stim.PauliString(n_qubits)
    if ps.sign == Sign.MINUS:
        pauli_str *= -1
    for node, axis in ps.axes.items():
        pauli_str[node] = axis.name
    return pauli_str


def stim_to_pauli_string(ps: stim.PauliString) -> tuple[PauliString, int]:
    """Transform a :class:`stim.PauliString` into a :class:`graphix.circ_ext.extraction.PauliString` instance.

    This function is the inverse of :func:`pauli_string_to_stim`.

    Parameters
    ----------
    stim.PauliString
        The Pauli string in `stim` format.

    Returns
    -------
    graphix.circ_ext.extraction.PauliString
        Converted Pauli string.
    int
        Width of the circuit on which the Pauli string is defined.

    Notes
    -----
    Qubits with the identity operator in the input ``stim.PauliString`` do not appear in the returned ``PauliString.axes.keys``.
    """
    axes: dict[int, Axis] = {}
    # ``stim.PauliString`` has no attribute ``__iter__``
    # (but it has ``__len__`` and ``__getitem__``)
    pauli: int
    for q, pauli in enumerate(ps):  # type: ignore[arg-type]
        match pauli:
            case 1:
                axes[q] = Axis.X
            case 2:
                axes[q] = Axis.Y
            case 3:
                axes[q] = Axis.Z

    sign = Sign.PLUS if ps.sign == 1 else Sign.MINUS

    return PauliString(axes, sign), len(ps)


def cm_to_stim_tableau(clifford_map: CliffordMap) -> stim.Tableau:
    """Transpile the Clifford map into a Stim Tableau.

    Parameters
    ----------
    clifford_map: CliffordMap
        The Clifford map to be transpiled.

    Returns
    -------
    stim.Tableau

    Raises
    ------
    NotImplementedError
        If ``len(self.input_nodes) != len(self.output_nodes)``.
    """
    if len(clifford_map.input_nodes) != len(clifford_map.output_nodes):
        raise NotImplementedError(
            ":class:`stim.Tableau` does not support isometries (Clifford maps with larger number of outputs than inputs)."
        )

    xs: list[stim.PauliString] = []
    zs: list[stim.PauliString] = []
    n_qubits = len(clifford_map.output_nodes)

    for qubit in range(n_qubits):
        xs.append(pauli_string_to_stim(clifford_map.x_map[qubit], n_qubits))
        zs.append(pauli_string_to_stim(clifford_map.z_map[qubit], n_qubits))

    return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)


def stim_tableau_to_cm(tab: stim.Tableau) -> CliffordMap:
    """Convert a Stim Tableau into a Clifford map representation.

    This function extracts the X and Z stabilizer mappings from a :class:`stim.Tableau`
    object and converts them into dictionaries mapping qubit indices to corresponding
    :class:`graphix.circ_ext.extraction.PauliString` objects. The resulting mappings are
    used to construct a :class:`graphix.circ_ext.extraction.CliffordMap` with identical
    input and output qubit ordering.

    Parameters
    ----------
    tab : stim.Tableau
        A Stim tableau representing a Clifford operation on ``n`` qubits.

    Returns
    -------
    graphix.circ_ext.extraction.CliffordMap

    Raises
    ------
    AssertionError
        If the number of qubits inferred from a Pauli string does not
        match the tableau size.
    """
    nqubits = len(tab)

    x_map: dict[int, PauliString] = {}
    z_map: dict[int, PauliString] = {}

    for mapping, stim_mapping in zip([x_map, z_map], [tab.x_output, tab.z_output], strict=True):
        for q in range(nqubits):
            mapping[q], n = stim_to_pauli_string(stim_mapping(q))
            assert nqubits == n

    qubit_list = list(range(nqubits))

    return CliffordMap(x_map=x_map, z_map=z_map, input_nodes=qubit_list, output_nodes=qubit_list)
