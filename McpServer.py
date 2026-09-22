import sys
from mcp.server import MCPServer

from CastorRag import retrieve_context

mcp = MCPServer("Castor Oil MCP Server")


@mcp.tool()
def search_castor_oil(query: str) -> str:
    """Search the Castor Oil PDF and return relevant information."""
    return retrieve_context(query)


@mcp.tool()
def unit_conversion(value: float, from_unit: str, to_unit: str) -> float:
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()

    conversions = {
        ("kg", "g"): value * 1000,
        ("g", "kg"): value / 1000,
        ("mg", "g"): value / 1000,
        ("g", "mg"): value * 1000,
        ("l", "ml"): value * 1000,
        ("ml", "l"): value / 1000,
        ("m", "cm"): value * 100,
        ("cm", "m"): value / 100,
        ("m", "mm"): value * 1000,
        ("mm", "m"): value / 1000,
    }

    key = (from_unit, to_unit)
    if key in conversions:
        return conversions[key]
    if from_unit == to_unit:
        return value
    raise ValueError(f"Conversion from {from_unit} to {to_unit} is not supported.")


@mcp.tool()
def calculate_oil_yield(
    weight_of_extracted_oil: float,
    initial_weight_of_raw_material: float,
) -> float:
    if initial_weight_of_raw_material <= 0:
        raise ValueError("Initial weight must be greater than zero.")
    return (weight_of_extracted_oil / initial_weight_of_raw_material) * 100


@mcp.tool()
def determination_of_specific_gravity(
    weight_of_oil_sample: float,
    weight_of_water: float,
) -> float:
    if weight_of_water <= 0:
        raise ValueError("Weight of water must be greater than zero.")
    return weight_of_oil_sample / weight_of_water


@mcp.tool()
def determination_of_kinematic_viscosity(
    dynamic_viscosity: float,
    density: float,
) -> float:
    if density <= 0:
        raise ValueError("Density must be greater than zero.")
    return dynamic_viscosity / density


@mcp.tool()
def determination_of_moisture_content(w1: float, w2: float) -> float:
    if w2 <= 0:
        raise ValueError("Final weight must be greater than zero.")
    return ((w1 - w2) / w2) * 100


if __name__ == "__main__":
    print("Starting Castor Oil MCP Server...", file=sys.stderr)
    mcp.run()