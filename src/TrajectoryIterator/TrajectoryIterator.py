"""
TrajectoryIterator: Single-pass trajectory iteration using Observer Pattern.

This module provides a TrajectoryIterator that iterates through a trajectory once
and notifies registered observers (subscribers) for each frame. This eliminates
the need to store coordinates in memory since MDAnalysis already provides them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import warnings

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise


class FrameObserver(ABC):
    """Abstract base class for frame observers in the Observer Pattern."""
    
    @abstractmethod
    def on_frame_start(self, iterator: 'TrajectoryIterator') -> None:
        """Called once before iteration starts."""
        pass
    
    @abstractmethod
    def on_frame(self, ts: mda.coordinates.base.Timestep, frame_idx: int, 
                 universe: mda.Universe) -> None:
        """
        Called for each frame during iteration.
        
        Args:
            ts: MDAnalysis Timestep object for the current frame
            frame_idx: Zero-based index of the current frame
            universe: MDAnalysis Universe (already positioned at current frame)
        """
        pass
    
    @abstractmethod
    def on_frame_end(self, iterator: 'TrajectoryIterator') -> None:
        """Called once after iteration completes."""
        pass
    
    def get_selections_needed(self) -> List[str]:
        """
        Return list of selection strings this observer needs.
        Used for optimization and validation.
        """
        return []


class TrajectoryIterator:
    """
    Single-pass trajectory iterator using Observer Pattern.
    
    Iterates through a trajectory once and notifies all registered observers
    for each frame. This eliminates redundant iterations and memory storage.
    
    Example:
        >>> iterator = TrajectoryIterator(u)
        >>> endpoint_observer = EndpointAnalyzerObserver(...)
        >>> volume_observer = VolumeAnalyzerObserver(...)
        >>> iterator.subscribe(endpoint_observer)
        >>> iterator.subscribe(volume_observer)
        >>> iterator.iterate()  # Single pass through trajectory
    """
    
    def __init__(self, universe: mda.Universe):
        """
        Initialize TrajectoryIterator.
        
        Args:
            universe: MDAnalysis Universe to iterate
        """
        self.universe = universe
        self.observers: List[FrameObserver] = []
        self.n_frames = len(universe.trajectory)
        self.frame_indices: List[int] = []
        self.times: List[float] = []
        self._iterated = False
    
    def subscribe(self, observer: FrameObserver) -> None:
        """
        Subscribe an observer to receive frame events.
        
        Args:
            observer: FrameObserver instance to subscribe
        """
        if observer not in self.observers:
            self.observers.append(observer)
    
    def unsubscribe(self, observer: FrameObserver) -> None:
        """
        Unsubscribe an observer from frame events.
        
        Args:
            observer: FrameObserver instance to unsubscribe
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def iterate(self, start: Optional[int] = None, stop: Optional[int] = None, 
                step: Optional[int] = None) -> None:
        """
        Iterate through trajectory once and notify all observers.
        
        Args:
            start: Start frame index (default: 0)
            stop: Stop frame index (default: n_frames)
            step: Step size (default: 1)
        """
        if self._iterated:
            warnings.warn(
                "TrajectoryIterator has already been iterated. "
                "Reset observers if you need to iterate again."
            )
        
        # Notify observers that iteration is starting
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Reset frame tracking
        self.frame_indices = []
        self.times = []
        
        # Iterate through trajectory
        frame_idx = 0
        for ts in self.universe.trajectory[start:stop:step]:
            self.frame_indices.append(ts.frame)
            self.times.append(ts.time)
            
            # Notify all observers
            for observer in self.observers:
                try:
                    observer.on_frame(ts, frame_idx, self.universe)
                except Exception as e:
                    warnings.warn(
                        f"Observer {type(observer).__name__} raised exception "
                        f"on frame {frame_idx}: {e}"
                    )
            
            frame_idx += 1
        
        # Notify observers that iteration is complete
        for observer in self.observers:
            observer.on_frame_end(self)
        
        self._iterated = True
    
    def get_frame_indices(self) -> List[int]:
        """Get list of frame indices from last iteration."""
        return self.frame_indices.copy()
    
    def get_times(self) -> List[float]:
        """Get list of times from last iteration."""
        return self.times.copy()
    
    def get_n_frames(self) -> int:
        """Get total number of frames in trajectory."""
        return self.n_frames
    
    def reset(self) -> None:
        """Reset iterator state (allows re-iteration)."""
        self._iterated = False
        self.frame_indices = []
        self.times = []

