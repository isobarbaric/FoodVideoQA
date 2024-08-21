import llm.frame_diff.frames as fd
from pathlib import Path

if __name__ == "__main__":
    fd.generate_frame_diff()
    fd.determine_eaten()
