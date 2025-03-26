import numpy as np
from typing import Dict, List, Optional, Tuple


Range = Tuple[int, int]


def sum_ranges(ranges: List[Range]):
    return sum(end - start for start, end in ranges)


def union_ranges(ranges: List[Range]) -> List[Range]:
    if ranges == []:
        return []
    
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged_ranges = [sorted_ranges[0]]
    
    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged_ranges[-1]
        
        if current_start <= last_end:
            merged_ranges[-1] = (last_start, max(last_end, current_end))
        else:
            merged_ranges.append((current_start, current_end))
    
    return merged_ranges


def intersect_ranges(range_1: Range, range_2: Range) -> Optional[Range]:
    start_1, end_1 = range_1
    start_2, end_2 = range_2
    
    start = max(start_1, start_2)
    end = min(end_1, end_2)
    
    return (start, end) if start <= end else None


def dict2range(d: Dict[str, int]) -> Range:
    return d['start_index'], d['end_index']


def calculate_metrics(
    retrieved_metadatas: List[List[Dict[str, int]]], 
    target_metadatas: List[List[Dict[str, int]]],
    metrics: List[str] = ['recall', 'precision', 'iou']
) -> Dict[str, float]:
    
    target_ranges_list = [
        [dict2range(range_dict) for range_dict in range_dicts] 
        for range_dicts in target_metadatas
    ]
    retrieved_ranges_list = [
        [dict2range(range_dict) for range_dict in range_dicts] 
        for range_dicts in retrieved_metadatas
    ]
    target_sizes, retrieved_sizes, intersections = [], [], []
    
    for target_ranges, retrieved_ranges in zip(target_ranges_list, retrieved_ranges_list):
        # calculate total number of tokens
        target_sizes += [sum_ranges(target_ranges)]
        retrieved_sizes += [sum_ranges(retrieved_ranges)]
        
        # calculate pairwise intersections
        question_intersections: List[Range] = []
        for target_range in union_ranges(target_ranges):
            for retrieved_range in union_ranges(retrieved_ranges):
                # ranges are sorted, no need to look further in retrieved ranges
                if retrieved_range[0] >= target_range[1]:
                    continue
                
                question_intersections += [
                    intersect_ranges(target_range, retrieved_range)
                ]
        
        # calculate total number of tokens on pairwise intersections
        # since ranges are disjoint (because of union_ranges), we can add them up
        question_intersections = [range for range in question_intersections if range is not None]
        intersections += [
            sum_ranges(question_intersections) if question_intersections else 0
        ]
    
    target_sizes = np.array(target_sizes, dtype=np.float32)
    retrieved_sizes = np.array(retrieved_sizes, dtype=np.float32)
    intersections = np.array(intersections, dtype=np.float32)
    
    metrics_funcs = {
        'recall': lambda: intersections / target_sizes * 100.,
        'precision': lambda: intersections / retrieved_sizes * 100.,
        'iou': lambda: intersections / (retrieved_sizes + target_sizes - intersections) * 100.
    }
    metrics_dict = {name: metrics_funcs[name]() for name in metrics}
    
    results = {}
    for name, values in metrics_dict.items():
        results[f'{name}_mean'] = np.mean(values).item()
        results[f'{name}_std'] = np.std(values).item()
    
    return results
