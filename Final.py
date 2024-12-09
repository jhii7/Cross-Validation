import subprocess

# Final.py runs both R2cpmpare.py and F1compare.py on the same input file and returns the results.

def main():
    filepath2 = input("Enter path to CSV: ")

    # R2compare.py
    print(" ")
    print(" ")
    print("R^2 Results:")
    subprocess.run(["python", "R2compare.py"], input=filepath2.encode())

    # F1compare.py
    print(" ")
    print(" ")
    print("F1 Results:")
    subprocess.run(["python", "F1compare.py"], input=filepath2.encode())

if __name__ == "__main__":
    main()
