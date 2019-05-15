import argparse

class CelephaisParser(argparse.ArgumentParser):
    def __init__(self):
        super(CelephaisParser, self).__init__()

        self.add_argument("--image", help="the image in which to detect faces", type=str)
