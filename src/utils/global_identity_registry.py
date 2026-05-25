"""
Global Identity Registry for PDF2Ontology Extraction Pipeline

This module provides centralized identity management across the three-stage
layered extraction chain, ensuring cross-modal consistency and preventing
duplicate entity creation.

"""

import re
import uuid
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IdentitySource(Enum):
    """Enumeration of identity sources across extraction stages"""
    STAGE1_TEXT = "stage1_text"
    STAGE2_VISUAL = "stage2_visual"
    STAGE3_FISSION = "stage3_fission"
    EXTERNAL = "external"


@dataclass
class EntityRecord:
    """
    Record of a registered entity in the global registry.
    
    Attributes:
        entity_id: Unique identifier (e.g., "proto_valve_3244")
        normalized_name: Normalized device/component name for matching
        original_name: Original name as it appeared in the document
        source: Which stage created this entity
        entity_type: Type of entity (Device, Component, etc.)
        metadata: Additional metadata for debugging/traceability
    """
    entity_id: str
    normalized_name: str
    original_name: str
    source: IdentitySource
    entity_type: str = "Unknown"
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"EntityRecord(id='{self.entity_id}', "
                f"name='{self.original_name}', source={self.source.value})")


class GlobalIdentityRegistry:
    """
    Centralized registry for managing entity identities across extraction stages.
    
    Key Responsibilities:
    1. Register entities from Stage 1 (text skeleton extraction)
    2. Resolve device IDs in Stage 2 (visual patching) by cross-referencing Stage 1
    3. Track dimension prototypes associated with each device
    4. Provide identity provenance information for debugging
    
    Design Principles:
    - Thread-safe for future concurrent processing
    - Deterministic ID generation for reproducibility
    - Fuzzy matching for robust device name resolution
    
    Example Usage:
        >>> registry = GlobalIdentityRegistry()
        >>> 
        >>> # Stage 1 completion: register prototypes
        >>> stage1_result = {"devices": [{"id": "proto_valve_3244", "model_name": "Type 3244"}]}
        >>> registry.register_from_stage1(stage1_result)
        >>> 
        >>> # Stage 2: resolve device ID from visual label
        >>> device_id = registry.resolve_device_id("Type 3244")
        >>> print(device_id)  # "proto_valve_3244" - reuses Stage 1 ID
    """
    
    def __init__(self):
        """Initialize the global identity registry"""
        # Core mapping: normalized_name -> entity_id
        self._name_to_id: Dict[str, str] = {}
        
        # Full entity records for provenance tracking
        self._entity_records: Dict[str, EntityRecord] = {}
        
        # Dimension prototypes grouped by device
        # device_id -> [dimension_prototype_dicts]
        self._device_dimensions: Dict[str, List[Dict]] = defaultdict(list)
        
        # Reverse lookup: entity_id -> normalized_name
        self._id_to_name: Dict[str, str] = {}
        
        # Track all registered names for debugging
        self._all_normalized_names: Set[str] = set()
        
        logger.info("GlobalIdentityRegistry initialized")
    
    # ========================
    # Core Registration Methods
    # ========================
    
    def register_from_stage1(self, stage1_result: Dict) -> None:
        """
        Register all entities from Stage 1 skeleton extraction.
        
        This method should be called after Stage 1 completes, before Stage 2 begins.
        It registers both device prototypes and any preliminary relationships.
        
        Args:
            stage1_result: Dictionary containing Stage 1 extraction results
                Expected structure:
                {
                    "devices": [
                        {
                            "id": "proto_valve_3244",
                            "label": "Type 3244 Valve",
                            "ontology_class": "ControlValve",
                            "attributes": [
                                {"property_name": "hasModelName", "value": "Type 3244"}
                            ],
                            ...
                        }
                    ],
                    "relationships": [...],  # Optional
                    ...
                }
        
        Example:
            >>> registry.register_from_stage1({
            ...     "devices": [
            ...         {"id": "proto_valve_3244", "label": "Type 3244"},
            ...         {"id": "proto_sensor_kn01", "label": "KN-01"}
            ...     ]
            ... })
            >>> len(registry.get_all_entities())
            2
        """
        devices = stage1_result.get("devices", [])
        
        registered_count = 0
        for device in devices:
            device_id = device.get("id")
            
            # Extract model name from multiple possible sources
            # Priority: 1) label field, 2) hasModelName attribute, 3) model_name field (legacy)
            model_name = device.get("label", "")
            
            if not model_name:
                # Try to extract from attributes
                attributes = device.get("attributes", [])
                for attr in attributes:
                    if attr.get("property_name") == "hasModelName":
                        model_name = attr.get("value", "")
                        break
            
            if not model_name:
                # Fallback to legacy field names
                model_name = device.get("model_name", device.get("name", ""))
            
            # Get entity class from ontology_class or fallback to class
            entity_class = device.get("ontology_class", device.get("class", "Unknown"))
            
            if not device_id or not model_name:
                logger.warning(f"Skipping device with missing id or model_name: {device}")
                continue
            
            # Register the device
            success = self._register_entity(
                entity_id=device_id,
                original_name=model_name,
                source=IdentitySource.STAGE1_TEXT,
                entity_type=entity_class,
                metadata={"stage1_data": device}
            )
            
            if success:
                registered_count += 1
        
        logger.info(f"Registered {registered_count} device prototypes from Stage 1")
    
    def register_dimension_prototype(
        self, 
        device_id: str, 
        dimension_data: Dict
    ) -> None:
        """
        Register a dimension prototype associated with a device.
        
        This is typically called during Stage 2 visual patching to record
        dimension prototypes that will be fissioned in Stage 3.
        
        Args:
            device_id: ID of the parent device prototype
            dimension_data: Dictionary containing dimension information
                Expected keys: "id", "parameter_name", "dimension_type", etc.
        
        Example:
            >>> registry.register_dimension_prototype(
            ...     device_id="proto_valve_3244",
            ...     dimension_data={
            ...         "id": "proto_valve_3244_dim_H1",
            ...         "parameter_name": "H1",
            ...         "dimension_type": "vertical",
            ...         "anchor_start_x": 100,
            ...         "anchor_start_y": 200
            ...     }
            ... )
        """
        if device_id not in self._entity_records:
            logger.warning(
                f"Cannot register dimension for unknown device '{device_id}'. "
                f"Dimension: {dimension_data.get('id', 'unknown')}"
            )
            return
        
        dim_id = dimension_data.get("id")
        if not dim_id:
            logger.warning(f"Dimension data missing 'id' field: {dimension_data}")
            return
        
        # Avoid duplicates
        existing_dim_ids = [d.get("id") for d in self._device_dimensions[device_id]]
        if dim_id in existing_dim_ids:
            logger.debug(f"Dimension '{dim_id}' already registered for device '{device_id}'")
            return
        
        self._device_dimensions[device_id].append(dimension_data)
        logger.debug(f"Registered dimension '{dim_id}' for device '{device_id}'")
    
    def _register_entity(
        self,
        entity_id: str,
        original_name: str,
        source: IdentitySource,
        entity_type: str = "Unknown",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Internal method to register a single entity.
        
        Returns:
            True if registration succeeded, False if duplicate
        """
        normalized_name = self._normalize_device_name(original_name)
        
        # Check for conflicts
        if normalized_name in self._name_to_id:
            existing_id = self._name_to_id[normalized_name]
            if existing_id != entity_id:
                logger.warning(
                    f"Name conflict detected: '{original_name}' (normalized: '{normalized_name}') "
                    f"already mapped to '{existing_id}', cannot remap to '{entity_id}'"
                )
                return False
            else:
                logger.debug(f"Entity '{entity_id}' already registered, skipping")
                return False
        
        # Create record
        record = EntityRecord(
            entity_id=entity_id,
            normalized_name=normalized_name,
            original_name=original_name,
            source=source,
            entity_type=entity_type,
            metadata=metadata or {}
        )
        
        # Register in all lookup structures
        self._name_to_id[normalized_name] = entity_id
        self._entity_records[entity_id] = record
        self._id_to_name[entity_id] = normalized_name
        self._all_normalized_names.add(normalized_name)
        
        logger.debug(f"Registered entity: {record}")
        return True
    
    # ========================
    # Identity Resolution Methods
    # ========================
    
    def resolve_device_id(
        self, 
        detected_name: str, 
        fallback_uuid: bool = True,
        source: IdentitySource = IdentitySource.STAGE2_VISUAL
    ) -> Optional[str]:
        """
        Resolve a device ID from a detected name (e.g., from visual label).
        
        Resolution Strategy:
        1. Normalize the detected name
        2. Check exact match in registered entities
        3. If fallback_uuid=True and no match found, generate new UUID
        4. If fallback_uuid=False and no match found, return None
        
        Args:
            detected_name: Raw device name/label from document (e.g., "Type 3244")
            fallback_uuid: Whether to generate a new UUID if no match found
            source: Source of this detection (for provenance tracking)
        
        Returns:
            Entity ID (existing or newly generated), or None if no match and fallback disabled
        
        Example:
            >>> # Assume "proto_valve_3244" was registered in Stage 1
            >>> registry.resolve_device_id("Type 3244")
            'proto_valve_3244'  # Reuses Stage 1 ID
            >>> 
            >>> registry.resolve_device_id("New Device 5000")
            'vis_new_device_5000_a3f2c1b9'  # Generates new UUID
            >>> 
            >>> registry.resolve_device_id("Unknown Device", fallback_uuid=False)
            None  # No fallback
        """
        normalized = self._normalize_device_name(detected_name)
        
        # Exact match
        if normalized in self._name_to_id:
            existing_id = self._name_to_id[normalized]
            logger.info(
                f"🔗 Cross-reference found: '{detected_name}' -> '{existing_id}' "
                f"(from {self._entity_records[existing_id].source.value})"
            )
            return existing_id
        
        # Fuzzy match (optional enhancement)
        fuzzy_match = self._fuzzy_match_device_name(normalized)
        if fuzzy_match:
            logger.info(
                f"🔗 Fuzzy match found: '{detected_name}' -> '{fuzzy_match}' "
                f"(normalized: '{normalized}' ~ '{self._id_to_name[fuzzy_match]}')"
            )
            return fuzzy_match
        
        # No match found
        if not fallback_uuid:
            logger.debug(f"No match found for '{detected_name}' and fallback disabled")
            return None
        
        # Generate new UUID
        new_id = self._generate_visual_uuid(normalized)
        self._register_entity(
            entity_id=new_id,
            original_name=detected_name,
            source=source,
            entity_type="Unknown",
            metadata={"auto_generated": True}
        )
        
        logger.info(f"🆕 New entity created: '{detected_name}' -> '{new_id}'")
        return new_id
    
    def _fuzzy_match_device_name(
        self, 
        normalized_query: str, 
        threshold: float = 0.85
    ) -> Optional[str]:
        """
        Attempt fuzzy matching against registered entity names.
        
        Uses simple string similarity (Levenshtein-like) to handle minor variations:
        - "type3244" vs "type_3244"
        - "kn01" vs "kn_01"
        
        Args:
            normalized_query: Normalized query string
            threshold: Similarity threshold (0-1)
        
        Returns:
            Entity ID of best match, or None if no good match found
        """
        if not self._all_normalized_names:
            return None
        
        from difflib import SequenceMatcher
        
        best_match = None
        best_ratio = 0.0
        
        for registered_name in self._all_normalized_names:
            ratio = SequenceMatcher(None, normalized_query, registered_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = registered_name
        
        if best_ratio >= threshold:
            return self._name_to_id[best_match]
        
        return None
    
    def _generate_visual_uuid(self, normalized_name: str) -> str:
        """
        Generate a deterministic UUID for visual-detected entities.
        
        Format: vis_{normalized_name}_{short_uuid}
        Example: "vis_type_5000_a3f2c1b9"
        """
        short_uuid = uuid.uuid4().hex[:8]
        return f"vis_{normalized_name}_{short_uuid}"
    
    # ========================
    # Query Methods
    # ========================
    
    def get_dimension_prototypes_for_device(self, device_id: str) -> List[Dict]:
        """
        Retrieve all dimension prototypes associated with a device.
        
        This is used in Stage 3 to fetch dimensions that need to be fissioned.
        
        Args:
            device_id: ID of the device prototype
        
        Returns:
            List of dimension prototype dictionaries
        
        Example:
            >>> dims = registry.get_dimension_prototypes_for_device("proto_valve_3244")
            >>> [d["parameter_name"] for d in dims]
            ['H1', 'L', 'd']
        """
        return self._device_dimensions.get(device_id, [])
    
    def get_entity_record(self, entity_id: str) -> Optional[EntityRecord]:
        """
        Retrieve full entity record by ID.
        
        Useful for debugging and provenance tracking.
        """
        return self._entity_records.get(entity_id)
    
    def get_all_entities(
        self, 
        source_filter: Optional[IdentitySource] = None
    ) -> List[EntityRecord]:
        """
        Get all registered entities, optionally filtered by source.
        
        Args:
            source_filter: If provided, only return entities from this source
        
        Returns:
            List of EntityRecord objects
        
        Example:
            >>> # Get all Stage 1 entities
            >>> stage1_entities = registry.get_all_entities(IdentitySource.STAGE1_TEXT)
            >>> [e.entity_id for e in stage1_entities]
            ['proto_valve_3244', 'proto_sensor_kn01']
        """
        if source_filter is None:
            return list(self._entity_records.values())
        
        return [
            record for record in self._entity_records.values()
            if record.source == source_filter
        ]
    
    def has_entity(self, entity_id: str) -> bool:
        """Check if an entity ID is registered"""
        return entity_id in self._entity_records
    
    def has_device_by_name(self, device_name: str) -> bool:
        """Check if a device with this name (normalized) is registered"""
        normalized = self._normalize_device_name(device_name)
        return normalized in self._name_to_id
    
    # ========================
    # Statistics & Debugging
    # ========================
    
    def get_statistics(self) -> Dict:
        """
        Get registry statistics for monitoring.
        
        Returns:
            Dictionary with counts by source, entity type, etc.
        """
        stats = {
            "total_entities": len(self._entity_records),
            "total_dimensions": sum(len(dims) for dims in self._device_dimensions.values()),
            "by_source": defaultdict(int),
            "by_entity_type": defaultdict(int),
            "devices_with_dimensions": len([
                device_id for device_id, dims in self._device_dimensions.items()
                if dims
            ])
        }
        
        for record in self._entity_records.values():
            stats["by_source"][record.source.value] += 1
            stats["by_entity_type"][record.entity_type] += 1
        
        return dict(stats)
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the registry state"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Global Identity Registry Summary")
        print("="*60)
        print(f"Total Entities: {stats['total_entities']}")
        print(f"Total Dimensions: {stats['total_dimensions']}")
        print(f"Devices with Dimensions: {stats['devices_with_dimensions']}")
        print("\nBy Source:")
        for source, count in stats['by_source'].items():
            print(f"  - {source}: {count}")
        print("\nBy Entity Type:")
        for entity_type, count in stats['by_entity_type'].items():
            print(f"  - {entity_type}: {count}")
        print("="*60 + "\n")
    
    def export_provenance_graph(self) -> Dict:
        """
        Export the full provenance graph for visualization/debugging.
        
        Returns:
            Dictionary suitable for JSON serialization with full entity records
        """
        return {
            "entities": {
                entity_id: {
                    "id": record.entity_id,
                    "name": record.original_name,
                    "normalized_name": record.normalized_name,
                    "source": record.source.value,
                    "type": record.entity_type,
                    "metadata": record.metadata
                }
                for entity_id, record in self._entity_records.items()
            },
            "dimension_associations": {
                device_id: [dim.get("id") for dim in dims]
                for device_id, dims in self._device_dimensions.items()
            },
            "statistics": self.get_statistics()
        }
    
    # ========================
    # Utility Methods
    # ========================
    
    @staticmethod
    def _normalize_device_name(raw_name: str) -> str:
        """
        Normalize device name for consistent matching.
        
        Normalization rules:
        1. Convert to lowercase
        2. Remove special characters (keep alphanumeric and spaces)
        3. Replace whitespace and hyphens with underscores
        4. Strip leading/trailing underscores
        
        Args:
            raw_name: Original device name from document
        
        Returns:
            Normalized name suitable for use as dictionary key
        
        Examples:
            >>> GlobalIdentityRegistry._normalize_device_name("Type 3244")
            'type_3244'
            >>> GlobalIdentityRegistry._normalize_device_name("KN-01-Series")
            'kn_01_series'
            >>> GlobalIdentityRegistry._normalize_device_name("Valve Ø25")
            'valve_25'
        """
        if not raw_name:
            return ""
        
        # Convert to lowercase
        normalized = raw_name.lower()
        
        # Remove common symbols (diameter, degree, etc.)
        normalized = re.sub(r'[øøØ∅φΦ°º]', '', normalized)
        
        # Keep only alphanumeric and whitespace/hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Replace whitespace and hyphens with underscores
        normalized = re.sub(r'[-\s]+', '_', normalized)
        
        # Strip leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def clear(self) -> None:
        """
        Clear all registry data.
        
        ⚠️ Use with caution! This is primarily for testing.
        """
        self._name_to_id.clear()
        self._entity_records.clear()
        self._device_dimensions.clear()
        self._id_to_name.clear()
        self._all_normalized_names.clear()
        logger.warning("Registry cleared - all entity records deleted")


# ========================
# Module-Level Convenience Functions
# ========================

def create_registry_from_stage1(stage1_result: Dict) -> GlobalIdentityRegistry:
    """
    Convenience function to create and populate a registry from Stage 1 results.
    
    Example:
        >>> registry = create_registry_from_stage1(stage1_extraction_output)
    """
    registry = GlobalIdentityRegistry()
    registry.register_from_stage1(stage1_result)
    return registry


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Global Identity Registry - Demo")
    print("="*60 + "\n")
    
    # Initialize registry
    registry = GlobalIdentityRegistry()
    
    # Simulate Stage 1 results (using correct structure)
    print("Step 1: Registering Stage 1 prototypes...")
    stage1_data = {
        "devices": [
            {
                "id": "proto_valve_3244",
                "label": "Type 3244 Control Valve",
                "ontology_class": "ControlValve",
                "attributes": [
                    {"property_name": "hasModelName", "value": "Type 3244"}
                ]
            },
            {
                "id": "proto_sensor_kn01",
                "label": "KN-01 Series Pressure Transmitter",
                "ontology_class": "PressureTransmitter",
                "attributes": [
                    {"property_name": "hasModelName", "value": "KN-01 Series"}
                ]
            }
        ]
    }
    registry.register_from_stage1(stage1_data)
    
    # Simulate Stage 2 resolution
    print("\nStep 2: Resolving device IDs in Stage 2...")
    test_cases = [
        "Type 3244 Control Valve",  # Exact match
        "Type 3244",  # Partial match
        "KN-01 Series",  # Exact match
        "Unknown Device 5000"  # New device
    ]
    
    for name in test_cases:
        resolved_id = registry.resolve_device_id(name)
        print(f"  '{name}' -> '{resolved_id}'")
    
    # Register dimensions
    print("\nStep 3: Registering dimension prototypes...")
    registry.register_dimension_prototype(
        "proto_valve_3244",
        {
            "id": "proto_valve_3244_dim_H1",
            "parameter_name": "H1",
            "dimension_type": "vertical"
        }
    )
    registry.register_dimension_prototype(
        "proto_valve_3244",
        {
            "id": "proto_valve_3244_dim_L",
            "parameter_name": "L",
            "dimension_type": "horizontal"
        }
    )
    
    # Query dimensions
    dims = registry.get_dimension_prototypes_for_device("proto_valve_3244")
    print(f"  Dimensions for 'proto_valve_3244': {[d['parameter_name'] for d in dims]}")
    
    # Print summary
    registry.print_summary()