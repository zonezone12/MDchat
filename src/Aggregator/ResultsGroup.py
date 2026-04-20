from typing import Dict, Any, Optional, List
import numpy as np

class ResultsGroup:
    """
    Declarative result aggregation for parallel processing.
    
    Inspired by MDAnalysis.analysis.results.ResultsGroup, this class provides
    a declarative way to specify how results from parallel workers should be
    merged back into the main observer.
    
    Usage:
        >>> aggregator = ResultsGroup(lookup={
        ...     'rows': ResultsGroup.list_extend_sorted('frame'),
        ...     'all_pairs': ResultsGroup.dict_merge_nonnan,
        ...     'frame_count': ResultsGroup.sum_values,
        ... })
        >>> aggregator.merge(main_results, worker_results)
    """
    
    def __init__(self, lookup: Optional[Dict[str, Any]] = None):
        """
        Initialize ResultsGroup with aggregation lookup.
        
        Args:
            lookup: Dictionary mapping result keys to aggregation functions.
                   Each function should have signature: (main_value, other_value) -> merged_value
        """
        self.lookup = lookup or {}
    
    def merge(self, main_results: Dict[str, Any], other_results: Dict[str, Any]) -> None:
        """
        Merge other_results into main_results using the lookup strategies.
        
        Args:
            main_results: The main results dictionary to merge into (modified in-place)
            other_results: The worker results dictionary to merge from
        """
        if other_results is None:
            return
        
        for key, other_value in other_results.items():
            if other_value is None:
                continue
            
            if key in self.lookup:
                aggregator = self.lookup[key]
                if key in main_results and main_results[key] is not None:
                    main_results[key] = aggregator(main_results[key], other_value)
                else:
                    # First value - just copy it
                    main_results[key] = other_value
            else:
                # No aggregator specified - use default behavior (overwrite)
                if key not in main_results or main_results[key] is None:
                    main_results[key] = other_value
    
    # ========== Aggregation Strategy Functions ==========
    
    @staticmethod
    def ndarray_vstack(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Stack arrays vertically (along axis 0)."""
        return np.vstack([main, other])
    
    @staticmethod
    def ndarray_hstack(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Stack arrays horizontally (along axis 1)."""
        return np.hstack([main, other])
    
    @staticmethod
    def ndarray_sum(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Sum arrays element-wise."""
        return main + other
    
    @staticmethod
    def ndarray_mean(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Average two arrays. Note: This is a simple average of two arrays,
        not a weighted mean based on number of frames.
        """
        return (main + other) / 2

    @staticmethod
    def ndarray_merge_nonnan(main: np.ndarray, other: np.ndarray) -> np.ndarray:
        """
        Merge same-shaped float arrays for parallel workers: copy non-NaN values
        from *other* into *main* (in-place) and return *main*.
        """
        if main.shape != other.shape:
            raise ValueError(
                f"ndarray_merge_nonnan shape mismatch: {main.shape} vs {other.shape}"
            )
        valid = ~np.isnan(other)
        main[valid] = other[valid]
        return main
    
    @staticmethod
    def list_extend(main: List, other: List) -> List:
        """Extend list with elements from other list."""
        result = list(main)
        result.extend(other)
        return result
    
    @staticmethod
    def list_extend_sorted(sort_key: str):
        """
        Return an aggregator that extends and sorts by a dictionary key.
        
        Args:
            sort_key: The key to sort dictionaries by (e.g., 'frame')
        
        Returns:
            Aggregation function that extends and sorts lists of dicts
        """
        def aggregator(main: List[Dict], other: List[Dict]) -> List[Dict]:
            result = list(main)
            result.extend(other)
            result.sort(key=lambda x: x.get(sort_key, 0))
            return result
        return aggregator
    
    @staticmethod
    def dict_merge_nonnan(main: Dict, other: Dict) -> Dict:
        """
        Merge dictionaries containing numpy arrays, keeping non-NaN values.
        
        For each key, if both dicts have arrays, copy non-NaN values from
        other into main where main has NaN values.
        """
        result = dict(main)
        for key, other_array in other.items():
            if key in result and result[key] is not None:
                main_array = result[key]
                if isinstance(main_array, np.ndarray) and isinstance(other_array, np.ndarray):
                    # Copy non-NaN values from other_array to main_array
                    valid_mask = ~np.isnan(other_array)
                    main_array[valid_mask] = other_array[valid_mask]
                    result[key] = main_array
                else:
                    # Non-array values - keep main value
                    pass
            else:
                # Key not in main - copy from other
                result[key] = other_array
        return result
    
    @staticmethod
    def min_value(main: Any, other: Any) -> Any:
        """Take the minimum value (useful for first_frame tracking)."""
        if main is None:
            return other
        if other is None:
            return main
        return min(main, other)
    
    @staticmethod
    def max_value(main: Any, other: Any) -> Any:
        """Take the maximum value (useful for last_frame tracking)."""
        if main is None:
            return other
        if other is None:
            return main
        return max(main, other)
    
    @staticmethod
    def sum_values(main: Any, other: Any) -> Any:
        """Sum scalar values (useful for counters)."""
        if main is None:
            return other
        if other is None:
            return main
        return main + other
    
    @staticmethod
    def flatten_sequence(main: List, other: List) -> List:
        """Flatten and concatenate sequences."""
        result = []
        for item in main:
            if isinstance(item, (list, tuple)):
                result.extend(item)
            else:
                result.append(item)
        for item in other:
            if isinstance(item, (list, tuple)):
                result.extend(item)
            else:
                result.append(item)
        return result
    
    @staticmethod
    def float_mean(main: float, other: float) -> float:
        """Average two float values."""
        if main is None:
            return other
        if other is None:
            return main
        return (main + other) / 2

