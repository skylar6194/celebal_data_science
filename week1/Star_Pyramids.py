##Create lower triangular, upper triangular and pyramid containing the "*" character.

def print_menu():
    print("\nStar Pattern Menu")
    print("------------------")
    print("1. Right Upper Triangular")
    print("2. Right Lower Triangular")
    print("3. Left Upper Triangular")
    print("4. Left Lower Triangular")
    print("5. Exit")

def print_right_upper_triangular(size):
    print("\nRight Upper Triangular Pattern:\n")
    for i in range(1, size + 1):
        print(" " * (i - 1) + "*" * (size - i + 1))

def print_right_lower_triangular(size):
    print("\nRight Lower Triangular Pattern:\n")
    for i in range(1, size + 1):
        print(" " * (size - i) + "*" * i)

def print_left_upper_triangular(size):
    print("\nLeft Upper Triangular Pattern:\n")
    for i in range(size, 0, -1):
        print("*" * i)

def print_left_lower_triangular(size):
    print("\nLeft Lower Triangular Pattern:\n")
    for i in range(1, size + 1):
        print("*" * i)

def main():
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice == '5':
            print("\nThank you for using the pattern printer by gaganjot\n")
            break
        if choice not in {'1', '2', '3', '4'}:
            print("\nInvalid choice. Please try again.\n")
            continue
        size_input = input("Enter the size of the pattern (positive integer): ").strip()
        if not size_input.isdigit() or int(size_input) <= 0:
            print("\nInvalid size entered. Please enter a positive integer.\n")
            continue
        size = int(size_input)
        if choice == '1':
            print_right_upper_triangular(size)
        elif choice == '2':
            print_right_lower_triangular(size)
        elif choice == '3':
            print_left_upper_triangular(size)
        elif choice == '4':
            print_left_lower_triangular(size)
        print()

if __name__ == "__main__":
    main()
