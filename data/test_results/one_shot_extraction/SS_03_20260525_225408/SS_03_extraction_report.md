# Extraction Report: SS_03

**PDF Pages Analyzed:** 4

**Resolution:** 300 DPI

**Base Namespace:** http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26#untitled-ontology-26#

---

## EXTRACTION REPORT

I model the assemblies as follows, respecting the given ontology:

- Type 3241 and Type 3244 are globe valves ⇒ subclasses of `:GlobeValve` (which already inherits from `:ValveBody`).  
- Type 3274 is an electrohydraulic actuator ⇒ subclass of `:ElectropneumaticActuatorComponent` (a kind of `:ValveActuator`).

Field-device level (conceptual control valves):
- `:ControlValve_3241_3274` – Type 3241/3274 Electric Control Valve.  
- `:ControlValve_3244_3274` – Type 3244/3274 Electric Control Valve.  

Each control valve instance has exactly one valve body and one actuator, satisfying the cardinality constraints.

Components:
- `:ValveBody_3241` and `:ValveBody_3244` represent generic valve-body families (all DN sizes).  
- Specific DN variants are represented as separate valve-body individuals with `hasNominalSizeDN`.  
- `:Actuator_3274_grp1` and `:Actuator_3274_grp2` represent the two actuator height variants from Tables 2.4 and 2.7.

Compatibility:
- All Type 3274 actuator variants are modeled as compatible with both Type 3241 and Type 3244 bodies via `hasCompatibleValveActuator` / `hasCompatibleValveBody` (inverse required by schema).

Dimensions:
- I only create `:Dimension` individuals for tables that are truly dimensional:
  - Table 2.1 – Type 3241 Valve (without actuator)
  - Table 2.2 – Type 3241 Valve with insulating section or bellows seal (without actuator)
  - Table 2.3 – Type 3241 Valve with heating jacket
  - Table 2.4 – Type 3274 Actuator
  - Table 2.5 – Type 3244 Valve (without actuator)
  - Table 2.6 – Type 3244 Valve with insulating section or bellows seal (without actuator)
  - Table 2.7 – Type 3274 Actuator (again, for Type 3244 valve; values same pattern as 2.4 but taken explicitly)

Non‑dimensional tables (Table 1.1 and 1.2 about KVS coefficients and Δp) are ignored.

Dimension orientation:
- From the drawings (Figures 3 and 4):
  - `L`, `L1`, `a`, `b`, `c` are horizontal.
  - `H`, `H1`, `H2`, `H3`, `H4`, `H5`, `H6`, `H9` are vertical.
- I encode this in `:hasDimensionType` with values `"horizontal"` or `"vertical"`.

Correction note:
- Initially I considered modeling each dimensional variant as a distinct `FieldDevice` individual. After reviewing that Tables 2.1–2.6 clearly apply to the valve bodies **without actuator**, I corrected this: dimensions are attached to valve-body or actuator components (`:ValveBody` / `:ValveActuator`), not to complete control valves. The final triples reflect this corrected modeling.

All numeric dimension values are taken directly from the tables; approximate weights are still treated as decimal numbers. Anchor coordinates are not given in the PDF, so `hasDimensionStartAnchorX`, `hasDimensionStartAnchorY`, `hasDimensionEndAnchorX`, and `hasDimensionEndAnchorY` are omitted (schema allows that).

---

## OWL/RDFS TRIPLES