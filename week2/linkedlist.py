class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def print_list(self):
        if self.head is None:
            print("The list is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        if self.head is None:
            print("Cannot delete from an empty list.")
            return

        if n < 1:
            print("Index must be a positive integer.")
            return

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        for i in range(n - 2):
            if current is None or current.next is None:
                print("Index out of range.")
                return
            current = current.next

        if current.next is None:
            print("Index out of range.")
            return

        current.next = current.next.next


def main():
    linked_list = LinkedList()

    while True:
        print("\nMenu:")
        print("1. Add a node to the end of the list")
        print("2. Print the list")
        print("3. Delete the nth node")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            data = input("Enter the data for the new node: ")
            linked_list.add_node(data)
        elif choice == '2':
            linked_list.print_list()
        elif choice == '3':
            n = int(input("Enter the position of the node to delete (1-based index): "))
            linked_list.delete_nth_node(n)
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
