import cProfile
import run

def main():
    run.main(1000)

if __name__ == "__main__":
    cProfile.run("main()", "Results/output.prof")