import time
import os


def main():
    start = time.time()
    for i in range(10):
        os.system("Start-Sleep -seconds 10")
        mid = time.time() - start
        print(f"step {i} at time {mid} seconds")

    end = time.time() - start
    print(f"end time {end} seconds")


if __name__ == "__main__":
    main()