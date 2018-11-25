import cv2
from pathlib import Path

def main():
    PGMPath = 'pgm/'
    JPGPath = 'jpg/'

    p = Path(PGMPath)
    p = sorted(p.glob("*.pgm"))

    for filename in p:
        print(filename)
        img = cv2.imread(PGMPath + filename.name, -1)
        cv2.imwrite(JPGPath + filename.name[:-3] + "jpg", img)

if __name__ == "__main__":
    main()