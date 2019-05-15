import sys

def fix_file(filename: str) -> None:
    """
    Take an MRC file which was written by MotionCor2 and fix it so that it has
    the correct string, ie.e. a space character, not a NUL character.
    """
    with open(filename, "r+b") as f:
        f.seek(208)
        map_id = f.read(4)
        if map_id == b"MAP\x00":
            f.seek(211)
            f.write(b"\x20")

if __name__ == "__main__":
    filenames = sys.argv[1:]

    for filename in filenames:
        fix_file(filename)
