import cProfile
import run

def main():
    run.main(10000)

if __name__ == "__main__":
    cProfile.run("main()", "Results/output.prof")