class FileWriterFactory:
    """
    A factory class to create and return a file writer object.
    """
    def __init__(self, filepath, mode='w'):
        """
        Initializes the FileWriterFactory with the file path and mode.

        Args:
            filepath (str): The path to the file.
            mode (str, optional): The mode in which to open the file ('w' for write, 'a' for append, etc.). Defaults to 'w'.
        """
        self.filepath = filepath
        self.mode = mode
        self.file_writer = None

    def get_file_writer(self):
        """
        Opens the file in the specified mode and returns the file writer object.

        Returns:
            _io.TextIOWrapper or None: The file writer object if the file is opened successfully,
                                       None otherwise.
        """
        try:
            self.file_writer = open(self.filepath, self.mode)
            return self.file_writer
        except Exception as e:
            print(f"Error opening file '{self.filepath}' in mode '{self.mode}': {e}")
            return None

    def close_file_writer(self):
        """
        Closes the file writer object if it's open.
        """
        if self.file_writer:
            try:
                self.file_writer.close()
                print(f"File '{self.filepath}' closed.")
                self.file_writer = None
            except Exception as e:
                print(f"Error closing file '{self.filepath}': {e}")

    def __enter__(self):
        """
        Allows the use of the class with the 'with' statement for automatic resource management.
        """
        self.file_writer = self.get_file_writer()
        return self.file_writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Handles the closing of the file writer when exiting the 'with' block.
        """
        self.close_file_writer()

if __name__ == "__main__":
    factory = FileWriterFactory("output.txt")

    # Method 1: Explicitly get the file writer and manage it
    writer = factory.get_file_writer()
    if writer:
        writer.write("This is the first line.\n")
        writer.write("Another line of text.\n")
        factory.close_file_writer()

    print("-" * 20)

    # Method 2: Using the 'with' statement for automatic management
    with FileWriterFactory("another_output.txt", mode='a') as writer:
        if writer:
            writer.write("Appending some data.\n")
            writer.write("More appended text.\n")
    # The file is automatically closed when exiting the 'with' block

    print("-" * 20)

    # You can also create the factory and get the writer in separate steps if needed
    factory_append = FileWriterFactory("yet_another.txt", 'a')
    appender = factory_append.get_file_writer()
    if appender:
        appender.write("Data written using a separate factory call.\n")
        factory_append.close_file_writer()