"""
File: config.py
Description: Global configuration for Reelpy. 
Users set once at the top of their script to apply defaults across all clips.
Example Usage:
    import reelpy
    reelpy.config.audio_mode = "full"
"""

class ReelpyConfig:
    def __init__(self):
        """
        audio_mode:
        -trim: default, audio trimmed to match video clip duration
        -extend: video freezes on last frame until audio finishes
        -full: audio plays for its full source duration, video ends when it ends
        """
        self.audio_mode: str = "trim"       
        self.default_bitrate: int = 4_000_000
        self.default_fps: float = 30.0

# single global instance, to be imported everywhere
config = ReelpyConfig()