"""Named Cypher templates for analytical graph queries.

The fallback repository uses the same logical template names while executing
in-memory filtering. Keeping query names here prevents string scattering in API
and router code.
"""

MATERIAL_OVERVIEW = "material_overview"
REGIME_OVERVIEW = "regime_overview"
PROPERTY_OVERVIEW = "property_overview"
MATERIAL_COMPARISON = "material_comparison"
REGIME_COMPARISON = "regime_comparison"
SIMILAR_EXPERIMENTS = "similar_experiments"
EQUIPMENT_USAGE = "equipment_usage"
LAB_ACTIVITY = "lab_activity"
GAP_ANALYSIS = "gap_analysis"
TOPIC_SEARCH = "topic_search"
GRAPH_NEIGHBORHOOD = "graph_neighborhood"


def template_for(name: str) -> str:
    """Return a stable template identifier used in diagnostics."""
    return name


def material_overview_query() -> str:
    """Cypher template for material overview queries."""
    return """
    MATCH (m:Material {canonical_name: $material})<-[:USES_MATERIAL]-(e:Experiment)
    OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
    OPTIONAL MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
    OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
    OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
    OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab:Laboratory)
    OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
    OPTIONAL MATCH (e)-[:HAS_GAP]->(gap:DataGap)
    OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
    RETURN e, collect(DISTINCT m) AS materials, collect(DISTINCT r) AS regimes,
           collect(DISTINCT {measurement: meas, property: p}) AS measurements,
           collect(DISTINCT eq) AS equipment, collect(DISTINCT team) AS teams,
           collect(DISTINCT lab) AS laboratories, collect(DISTINCT concl) AS conclusions,
           collect(DISTINCT gap) AS gaps, collect(DISTINCT chunk) AS chunks,
           collect(DISTINCT doc) AS documents
    """


def regime_overview_query() -> str:
    """Cypher template for process regime overview queries."""
    return """
    MATCH (r:ProcessRegime {canonical_name: $regime})<-[:HAS_REGIME]-(e:Experiment)
    OPTIONAL MATCH (e)-[:USES_MATERIAL]->(m:Material)
    OPTIONAL MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
    OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
    RETURN e, collect(DISTINCT m) AS materials, collect(DISTINCT r) AS regimes,
           collect(DISTINCT {measurement: meas, property: p}) AS measurements,
           collect(DISTINCT chunk) AS chunks, collect(DISTINCT doc) AS documents
    """


def property_overview_query() -> str:
    """Cypher template for property overview queries."""
    return """
    MATCH (p:Property {canonical_name: $property})<-[:OF_PROPERTY]-(meas:Measurement)<-[:MEASURED]-(e:Experiment)
    OPTIONAL MATCH (e)-[:USES_MATERIAL]->(m:Material)
    OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
    OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
    RETURN e, collect(DISTINCT m) AS materials, collect(DISTINCT r) AS regimes,
           collect(DISTINCT {measurement: meas, property: p}) AS measurements,
           collect(DISTINCT chunk) AS chunks, collect(DISTINCT doc) AS documents
    """


def gap_analysis_query() -> str:
    """Cypher template for constraint-filtered DataGap queries."""
    return """
    MATCH (g:DataGap)
    OPTIONAL MATCH (g)-[:GAP_FOR_ENTITY]->(m:Material)
    OPTIONAL MATCH (g)-[:GAP_FOR_REGIME]->(r:ProcessRegime)
    OPTIONAL MATCH (g)-[:GAP_FOR_PROPERTY]->(p:Property)
    WHERE ($material IS NULL OR m.canonical_name = $material OR g.material = $material)
      AND ($regime IS NULL OR r.canonical_name = $regime OR g.regime = $regime)
      AND ($property IS NULL OR p.canonical_name = $property OR g.property = $property)
    RETURN g, collect(DISTINCT m) AS materials, collect(DISTINCT r) AS regimes,
           collect(DISTINCT p) AS properties
    """
