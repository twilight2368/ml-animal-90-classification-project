def read_class_names_txt(PATH_TO_TXT_FILE):

    # Open the file and read its content
    with open(PATH_TO_TXT_FILE, 'r') as file:
        content = file.read()

    # # Print the content
    # print(content)

    class_names = content.split("\n")

    print("Number of classes:", len(class_names))
    print(class_names)
