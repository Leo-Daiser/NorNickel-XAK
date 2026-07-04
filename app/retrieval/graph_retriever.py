"""Small adapter between QueryConstraints and GraphRepository."""

from __future__ import annotations

from ..domain.query_constraints import QueryConstraints
from ..graph.graph_models import GraphQueryResult
from ..graph.graph_repository import GraphRepository


class GraphRetriever:
    """Run structured graph queries from parsed query constraints."""

    def __init__(self, repository: GraphRepository) -> None:
        self.repository = repository

    def material_regime_property(self, constraints: QueryConstraints) -> GraphQueryResult:
        material = constraints.materials[0]
        regime = constraints.regimes[0]
        property_name = constraints.properties[0]
        exact = self.repository.find_exact_material_regime_property(material, regime, property_name)
        partial = self.repository.find_partial_matches(material=material, regime=regime, property_name=property_name)
        gaps = self.repository.find_gaps(material=material, regime=regime, property_name=property_name)
        return GraphQueryResult(exact=exact, partial_matches=partial, gaps=gaps)

