"""Cypher snippets for the Neo4j-backed strict graph repository."""

EXACT_MATERIAL_REGIME_PROPERTY = """
MATCH (m:Material)<-[:USES_MATERIAL]-(e:Experiment)
MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
WHERE m.canonical_name = $material
  AND r.canonical_name = $regime
  AND p.canonical_name = $property
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WHERE coalesce(doc.active, true) = true
  AND coalesce(chunk.active, true) = true
RETURN e,
       collect(DISTINCT m) AS materials,
       collect(DISTINCT r) AS regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
"""

FIND_EXPERIMENTS_BY_CONSTRAINTS = """
MATCH (e:Experiment)
OPTIONAL MATCH (e)-[:USES_MATERIAL]->(m:Material)
WITH e, collect(DISTINCT m) AS materials
WHERE $material IS NULL OR any(item IN materials WHERE item.canonical_name = $material)
OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
WITH e, materials, collect(DISTINCT r) AS regimes
WHERE $regime IS NULL OR any(item IN regimes WHERE item.canonical_name = $regime)
MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
WHERE $property IS NULL OR p.canonical_name = $property
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WHERE coalesce(doc.active, true) = true
  AND coalesce(chunk.active, true) = true
RETURN e,
       materials,
       regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
LIMIT $limit
"""

DECISION_HISTORY_BY_MATERIAL = """
MATCH (m:Material)<-[:USES_MATERIAL]-(e:Experiment)
WHERE m.canonical_name = $material
OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
OPTIONAL MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WHERE coalesce(doc.active, true) = true
  AND coalesce(chunk.active, true) = true
RETURN e,
       collect(DISTINCT m) AS materials,
       collect(DISTINCT r) AS regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
"""

FIND_GAPS = """
MATCH (g:DataGap)
OPTIONAL MATCH (g)-[:GAP_FOR_ENTITY]->(m:Material)
OPTIONAL MATCH (g)-[:GAP_FOR_REGIME]->(r:ProcessRegime)
OPTIONAL MATCH (g)-[:GAP_FOR_PROPERTY]->(p:Property)
MATCH (g)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WHERE coalesce(doc.active, true) = true
  AND coalesce(chunk.active, true) = true
WITH g,
     collect(DISTINCT m) AS materials,
     collect(DISTINCT r) AS regimes,
     collect(DISTINCT p) AS properties,
     collect(DISTINCT chunk) AS chunks,
     collect(DISTINCT doc) AS documents
WHERE ($material IS NULL OR any(item IN materials WHERE item.canonical_name = $material) OR g.material = $material)
  AND ($regime IS NULL OR any(item IN regimes WHERE item.canonical_name = $regime) OR g.regime = $regime)
  AND ($property IS NULL OR any(item IN properties WHERE item.canonical_name = $property) OR g.property = $property)
RETURN g,
       materials,
       regimes,
       properties,
       chunks,
       documents
ORDER BY g.gap_id
"""

FIND_ACCEPTED_FACTS = """
MATCH (f:AcceptedFact)
WHERE coalesce(f.validation_status, 'accepted') = 'accepted'
  AND ($fact_types IS NULL OR f.fact_type IN $fact_types)
MATCH (f)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WHERE coalesce(doc.active, true) = true
  AND coalesce(chunk.active, true) = true
OPTIONAL MATCH (f)-[:FACT_SUBJECT|USES_MATERIAL]->(m:Material)
OPTIONAL MATCH (f)-[:FACT_PROCESS]->(proc:ProcessRegime)
OPTIONAL MATCH (f)-[:FACT_PARAMETER]->(prop:Property)
OPTIONAL MATCH (f)-[:USES_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (f)-[:FACT_SUBJECT]->(facility:Facility)
OPTIONAL MATCH (f)-[:HAS_GEOGRAPHY]->(geo:Geography)
OPTIONAL MATCH (f)-[:DESCRIBED_IN]->(pub:Publication)
OPTIONAL MATCH (f)-[:HAS_EXPERT]->(expert:Employee)
OPTIONAL MATCH (f)-[:PERFORMED_AT]->(lab:Laboratory)
OPTIONAL MATCH (f)-[:PERFORMED_BY]->(team:ResearchTeam)
RETURN f,
       collect(DISTINCT m) AS materials,
       collect(DISTINCT proc) AS processes,
       collect(DISTINCT prop) AS properties,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT facility) AS facilities,
       collect(DISTINCT geo) AS geographies,
       collect(DISTINCT pub) AS publications,
       collect(DISTINCT expert) AS experts,
       collect(DISTINCT lab) AS laboratories,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY f.confidence DESC, f.fact_id
LIMIT $limit
"""
