from wasp39.main import main
import sys


def main():
    # Pretend we ran: python -m wasp39.main --steps 0,1,4 --skip-spectrum-mcmc
    sys.argv = [
        "python -m wasp39.main",
        #"--steps", "0,1,2,3,4",
        #"--steps", "0,1,2,3",
        #"--steps", "0,1,2"
        "--steps", "0,6"
    ]

    from wasp39.main import main as pipeline_main
    pipeline_main()

if __name__ == "__main__":
    main()