import numpy as np
import basis_set_exchange as bse

from basis import contracted_gto

_STR_TO_PRIMITIVE_TYPE = {
    "gto": contracted_gto.PrimitiveType.CARTESIAN,
    "gto_cartesian": contracted_gto.PrimitiveType.CARTESIAN,
    "gto_spherical": contracted_gto.PrimitiveType.SPHERICAL,
}


def load(basis_name: str, element: int) -> list[contracted_gto.ContractedGTO]:
    """Loads contracted GTOs for a given element from the Basis Set Exchange."""
    bse_data = bse.get_basis(basis_name, elements=[element])
    electron_shells = bse_data["elements"][str(element)]["electron_shells"]

    contracted_gtos = []
    for shell in electron_shells:
        angular_momentum = shell["angular_momentum"]
        if len(angular_momentum) == 1:
            angular_momentum *= len(shell["coefficients"])

        contracted_gtos.append(
            contracted_gto.ContractedGTO(
                primitive_type=_STR_TO_PRIMITIVE_TYPE[shell["function_type"]],
                angular_momentum=tuple(angular_momentum),
                exponents=np.array(shell["exponents"], dtype=np.float64),
                coefficients=np.array(shell["coefficients"], dtype=np.float64),
            )
        )

    return contracted_gtos
