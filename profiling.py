import cProfile

import reinforcing

def main():
    reinforcing.main(1000)

if __name__ == "__main__":
    cProfile.run("main()", "output.prof")