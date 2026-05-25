# Extraction Report: EH_01

**PDF Pages Analyzed:** 36

**Resolution:** 300 DPI

**Base Namespace:** http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26#untitled-ontology-26#

---

## EXTRACTION REPORT

From the Proline Promag 10P datasheet, I identified:

- A **measuring system** consisting of:
  - a **transmitter “Promag 10”** (remote version drawing with dimensions on p.20),
  - an **electromagnetic flow sensor “Promag P”** (remote version, p.24),
  - and for the **compact version** these are combined into one housing (p.22).

Under the provided ontology, the closest superclass for the complete instrument is `Flowmeter` (subclass of `MeasuringSystem`). The internal separation into *transmitter* and *sensor* is described textually, but the ontology does not define specific subclasses for these, so I model them as `DeviceComponent` instances attached to the `Flowmeter` via `hasDeviceComponent`.

### Important correction

Initially, when seeing separate transmitter drawings, I considered modeling “Promag 10 transmitter” as a standalone `Flowmeter`.  
**My previous conclusion about "Promag 10 as its own Flowmeter device" is no longer accurate.**  
Later pages (notably p.3 “Measuring system consists of a transmitter and a sensor” and the dimensional tables grouping variants under “Promag 10P”) clarified that the sales/mechanical unit is the Promag 10P measuring system with compact and remote variants.

**Correction strategy and final modeling:**

- Introduce a main instance `:Flowmeter_Promag10P` of class `Flowmeter` with two variant components:
  - `:Transmitter_Promag10_Remote` (DeviceComponent, remote transmitter).
  - `:Sensor_PromagP_Remote` (DeviceComponent, remote sensor body).
- For compact-version dimension tables (p.22–23), instead of a separate Flowmeter, I treat the **compact flowmeter body** as another `DeviceComponent` of the same main Flowmeter, because the ontology does not separate “flowmeter body” as a ValveBody-like concept.
- All geometric data is modeled as instances of `Dimension` linked by `hasDimension`.

### Dimensional tables extracted

The following tables are clearly **dimensional** and were fully extracted:

1. **Transmitter, remote version – dimensions (p.20)**  
   - SI units (mm): parameters A, B, C, D, E, F, G, ØH, J, K, L, M, N, O, P.  
   - US units (inch): A, B, C, D, E, F, G, ØH, J, K, L, M, N, O, P.

2. **Compact version – flowmeter body (p.22–23)**  
   - For each nominal size DN 25–600 (EN/JIS/AS and ANSI tables) in mm and inch.  
   - Parameters: L, A, B, C, D, E, F, G, H, J, K (L varies with DN; A–K largely constant but given; I is missing in tables, so not modeled).

3. **Sensor, remote version (p.24–25)**  
   - Similar DN-based tables in mm and inch.  
   - Parameters: L, A, B, C, D, E, F, G, H, J.

4. **Ground disk for flange connections (p.26–27)**  
   - DN-based tables for the ground disk accessory.  
   - Parameters: A, B, C, D, E, t.  
   - These are modeled as dimensions of a dedicated `DeviceComponent` (ground disk).

All of these are about physical geometry, with numeric values as table data; performance tables (Kv, flow, pressure, etc.) were **ignored** as required.

### Dimension orientation

For orientation (`hasDimensionType`):

- I inspected the drawings:
  - On p.20, A, B, C, D, E, F, G, J, K, L, M, N, O, P are along horizontal or vertical dimension lines. However, the drawing is fairly complex and some axes are not clearly distinguishable from the reproduction; to avoid misclassification, I conservatively set `hasDimensionType "other"` for all parameters from this sheet.
  - On p.22, p.24, p.26 drawings likewise have multiple orientations but labels are not unambiguously horizontal or vertical for all; I again use `"other"`.
- No explicit diameter symbol (ø) is used in the tables, so none are marked `"diameter"`.

### Instance structure overview

- `:Flowmeter_Promag10P` (Flowmeter)  
  - `:hasDeviceComponent`:
    - `:Transmitter_Promag10_Remote`
    - `:Sensor_PromagP_Remote`
    - `:FlowmeterBody_Compact`
    - `:GroundDisk_ForFlangeConnections`
- Each component has multiple `Dimension` instances.  
- `hasModelName` is set on all `FieldDevice` and `DeviceComponent` individuals.

Note: The ontology provides valve-related classes and `hasCompatibleValveActuator` / `hasCompatibleValveBody` which are **not applicable** to this flowmeter. To respect domain and range constraints, I **do not create any instances or triples using those properties**; they would be invalid here.

## OWL/RDFS TRIPLES