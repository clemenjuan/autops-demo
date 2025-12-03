"""
Semantic Memory Module for CoALA Architecture

Long-term storage of factual knowledge about satellite operations,
regions, constraints, and domain knowledge.
"""

from typing import Dict, List, Any, Optional
from agent.memory.base_memory import BaseMemory


class SemanticMemory(BaseMemory):
    """
    Semantic Memory: Long-term factual knowledge.
    
    Stores domain knowledge such as:
    - Geographic information (Bayern coordinates, Munich bbox, Alps location)
    - Object detection facts (typical vehicle counts, ship patterns)
    - Operational constraints (weather conditions, satellite coverage)
    - Domain-specific knowledge
    
    Organized by concepts/entities for efficient retrieval.
    Future: Implement vector embeddings for semantic similarity search.
    """
    
    def __init__(self, file_path: str = "data/memory/semantic_memory.toon"):
        """
        Initialize semantic memory.
        
        Args:
            file_path: Path to TOON file for persistence
        """
        super().__init__(memory_type="semantic", file_path=file_path, persistent=True)
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """Initialize with some default domain knowledge if memory is empty."""
        if len(self.data) == 0:
            default_facts = [
                {
                    'concept': 'region',
                    'entity': 'Bayern',
                    'fact_type': 'location',
                    'content': 'Bayern (Bavaria) is the largest state in Germany, located in the southeast',
                    'tags': ['germany', 'europe', 'bayern', 'bavaria'],
                    'source': 'default_initialization'
                },
                {
                    'concept': 'region',
                    'entity': 'Munich',
                    'fact_type': 'location',
                    'content': 'Munich (München) is the capital of Bayern, coordinates approximately [48.1351, 11.5820]',
                    'tags': ['munich', 'münchen', 'bayern', 'city', 'capital'],
                    'source': 'default_initialization'
                },
                {
                    'concept': 'region',
                    'entity': 'Alps',
                    'fact_type': 'geography',
                    'content': 'The Alps mountain range forms the southern border of Bayern',
                    'tags': ['alps', 'mountains', 'bayern', 'geography'],
                    'source': 'default_initialization'
                },
                {
                    'concept': 'region',
                    'entity': 'Isar',
                    'fact_type': 'geography',
                    'content': 'The Isar river flows through Munich from south to north',
                    'tags': ['isar', 'river', 'munich', 'water'],
                    'source': 'default_initialization'
                },
                {
                    'concept': 'constraint',
                    'entity': 'cloud_cover',
                    'fact_type': 'weather',
                    'content': 'Bayern typically has moderate cloud cover; Alps region often has higher cloud coverage',
                    'tags': ['weather', 'clouds', 'alps', 'bayern'],
                    'source': 'default_initialization'
                }
            ]
            
            for fact in default_facts:
                self.store(fact)
    
    def store(self, entry: Dict[str, Any]) -> str:
        """
        Store a fact in semantic memory.
        
        Args:
            entry: Fact dictionary with keys:
                - concept: High-level concept (e.g., 'region', 'constraint', 'object_type')
                - entity: Specific entity (e.g., 'Munich', 'vehicle', 'cloud_cover')
                - fact_type: Type of fact (e.g., 'location', 'property', 'relation')
                - content: The actual factual content
                - tags: List of tags for retrieval
                - source: Where this fact came from
                - confidence: Optional confidence score
            
        Returns:
            entry_id: Unique identifier for the stored fact
        """
        entry = self._add_timestamp(entry)
        if 'id' not in entry:
            entry['id'] = self._generate_id()
        
        if 'concept' not in entry or 'content' not in entry:
            raise ValueError("Semantic entry must include 'concept' and 'content' fields")
        
        if 'tags' not in entry:
            entry['tags'] = []
        
        self.data.append(entry)
        
        if self.persistent:
            self.save()
        
        return entry['id']
    
    def retrieve(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve facts based on concept, entity, or tags.
        
        Args:
            query: Query parameters with keys like:
                - concept: Concept to search for
                - entity: Entity to search for
                - tags: List of tags to match
                - fact_type: Type of fact
                - keywords: List of keywords to search in content
            limit: Maximum number of facts to return
            
        Returns:
            List of matching facts
            
        Note: Currently uses keyword/tag matching.
        Future: Semantic similarity search using embeddings.
        """
        results = []
        concept = query.get('concept', '').lower()
        entity = query.get('entity', '').lower()
        tags = [t.lower() for t in query.get('tags', [])]
        fact_type = query.get('fact_type', '').lower()
        keywords = [k.lower() for k in query.get('keywords', [])]
        
        for fact in self.data:
            score = 0
            
            if concept and concept == fact.get('concept', '').lower():
                score += 3
            
            if entity and entity == fact.get('entity', '').lower():
                score += 3
            
            if fact_type and fact_type == fact.get('fact_type', '').lower():
                score += 1
            
            fact_tags = [t.lower() for t in fact.get('tags', [])]
            for tag in tags:
                if tag in fact_tags:
                    score += 2
            
            content = fact.get('content', '').lower()
            for keyword in keywords:
                if keyword in content:
                    score += 1
            
            if score > 0 or (not concept and not entity and not tags and not keywords):
                results.append({
                    'fact': fact,
                    'relevance_score': score
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return [r['fact'] for r in results[:limit]]
    
    def get_by_concept(self, concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all facts related to a concept."""
        return self.retrieve({'concept': concept}, limit=limit)
    
    def get_by_entity(self, entity: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all facts about a specific entity."""
        return self.retrieve({'entity': entity}, limit=limit)
    
    def get_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Get facts matching any of the provided tags."""
        return self.retrieve({'tags': tags}, limit=limit)
    
    def store_region_info(self, region_name: str, bbox: List[float], center: List[float], metadata: Dict[str, Any] = None) -> str:
        """
        Helper method to store geographic region information.
        
        Args:
            region_name: Name of the region (e.g., 'Munich', 'Bayern')
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            center: Center coordinates [lat, lon]
            metadata: Additional metadata
            
        Returns:
            entry_id: ID of stored fact
        """
        tags = ['region', 'geography', region_name.lower()]
        if metadata:
            tags.extend(metadata.get('tags', []))
        
        return self.store({
            'concept': 'region',
            'entity': region_name,
            'fact_type': 'bounding_box',
            'content': f'{region_name} bounding box: {bbox}, center: {center}',
            'data': {
                'bbox': bbox,
                'center': center,
                'metadata': metadata or {}
            },
            'tags': tags,
            'source': 'region_mapper_tool'
        })
    
    def store_detection_result(self, region: str, object_type: str, count: int, confidence: float, metadata: Dict[str, Any] = None) -> str:
        """
        Helper method to store object detection results as facts.
        
        Args:
            region: Region where detection occurred
            object_type: Type of object detected
            count: Number of objects detected
            confidence: Detection confidence
            metadata: Additional metadata
            
        Returns:
            entry_id: ID of stored fact
        """
        return self.store({
            'concept': 'detection',
            'entity': f'{object_type}_in_{region}',
            'fact_type': 'count',
            'content': f'Detected {count} {object_type} in {region} with confidence {confidence}',
            'data': {
                'region': region,
                'object_type': object_type,
                'count': count,
                'confidence': confidence,
                'metadata': metadata or {}
            },
            'tags': ['detection', object_type, region.lower()],
            'source': 'object_detector_tool',
            'confidence': confidence
        })
    
    def get_all_concepts(self) -> List[str]:
        """Get list of all unique concepts in memory."""
        concepts = set()
        for fact in self.data:
            concept = fact.get('concept')
            if concept:
                concepts.add(concept)
        return sorted(list(concepts))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about semantic memory."""
        if not self.data:
            return {
                'total_facts': 0,
                'concepts': [],
                'most_common_tags': []
            }
        
        tag_counts = {}
        for fact in self.data:
            for tag in fact.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        most_common = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_facts': len(self.data),
            'concepts': self.get_all_concepts(),
            'most_common_tags': [{'tag': tag, 'count': count} for tag, count in most_common]
        }

