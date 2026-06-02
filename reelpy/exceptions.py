"""
File: exceptions.py
Description: Small exception package for wrapping third-party errors
Notes: All errors are members of Python's 'Exception' class structure
"""

class ReelpyError(Exception):
    """Base exception for all Reelpy errors"""
    pass

class InvalidVideoError(ReelpyError):
    """Raised when a file cannot be opened as a video"""
    pass

class StreamNotFoundError(ReelpyError):
    """Raised when no video or audio stream exists in the file"""
    pass
