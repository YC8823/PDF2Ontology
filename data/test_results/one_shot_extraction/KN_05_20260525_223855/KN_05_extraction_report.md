# Extraction Report: KN_05

**PDF Pages Analyzed:** 24

**Resolution:** 300 DPI

**Base Namespace:** http://www.semanticweb.org/yanha/ontologies/2025/7/untitled-ontology-26#untitled-ontology-26#

---

## EXTRACTION REPORT

Initial modeling and assumptions
- I started by assuming the OPTIBAR PSM 2010 is a single “Transmitter” device. However, the ontology constrains Transmitter to be a DeviceComponent, while complete instruments must be modeled as FieldDevice subclasses.
- Correction: My previous conclusion about “the complete device being a Transmitter” is no longer accurate. I revised the model so that the complete instrument is a MeasuringSystem (a FieldDevice), which contains a Transmitter component, an ElectricalConnector (M12x1, 4‑pin), a Housing (display/indicator head), optional Cooling Fins, and several ProcessConnection variants (threads, clamp, Varivent).

Dimension interpretation corrections
- Initially, I considered assigning dimension “a, b, c” from Fig. 2‑1 to the general housing. After reading the notes below Fig. 2‑2 (“The total length of the device consists of the electrical connection (a), the transmitter housing (e) and the process connection (k).”), I corrected this:
  - a → belongs to the electrical connection (connector)
  - e → belongs to the transmitter housing
  - k (and m, n) → belong to the process connection variants

Variant strategy
- Process connections are modular; each variant (six threaded, three clamp DNs, one Varivent) receives its own ProcessConnection instance.
- Where the dimensional table splits values by DN (Clamp DN25/DN38/DN51), I created separate instances and separate Dimension individuals per DN.
- I classified dimension orientation by reading the arrows in the figures:
  - a, b, e, k, m, n, r → vertical
  - c → horizontal
  - f and some h values marked with “ø” → diameter
  - g (WSxx across flats) and h (thread designations) textual → other

Tables used (dimensional only)
- Fig. 2‑1 Dimensions (a, b, c)
- Fig. 2‑2 Dimensions for threaded process connections ①…⑥ (e, f, g, h, k, m, n)
- Fig. 2‑3 Dimensions for cooling fins and hygienic process connections ①…③ (e, f, h, k, r)

Non‑dimensional tables (measuring ranges, electrical specs, approvals) were ignored per instructions.

Final structure
- One MeasuringSystem instance for the OPTIBAR PSM 2010 device.
- Device components: Transmitter, ElectricalConnector (M12x1 4‑pin), Housing (display/indicator head), optional CoolingFins (as Housing), and 10 ProcessConnection variants (6 threaded + 3 clamp DNs + 1 Varivent).
- Dimensions linked to the appropriate component with hasDimension, using decimal mm values and, where provided, inch equivalents via hasDimensionStringValue. Orientation captured using hasDimensionType.

## OWL/RDFS TRIPLES