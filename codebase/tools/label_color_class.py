# label_colors.py

# Define the classes
classes = (
    'Box',
    'Table',
    'Chair',
    'Others',
    'Building',
    'Computer',
    'People',
)

# Define the corresponding colors as RGB tuples
palette = [
    (220, 20, 60),  # Box
    (119, 11, 32),  # Table
    (0, 0, 142),   # Chair
    (0, 0, 230),   # Others
    (106, 0, 228), # Building
    (34, 11, 67),  # Computer
    (160, 122, 2), # People
]

# Create a dictionary that maps class names to colors
class_to_color = {classes[i]: palette[i] for i in range(len(classes))}

# Create a dictionary that maps labels (both integer and binary form) to class names
label_to_class = {i: classes[i] for i in range(len(classes))}
label_to_class.update({tuple([int(x) for x in format(i, '03b')]): classes[i] for i in range(len(classes))})

# Create a dictionary that maps labels (both integer and binary form) to colors
label_to_color = {i: palette[i] for i in range(len(classes))}
label_to_color.update({tuple([int(x) for x in format(i, '03b')]): palette[i] for i in range(len(classes))})


# Export the class names, class-to-color, and label-to-class and label-to-color mappings for external use
if __name__ == "__main__":
    data = {
        "classes": classes,
        "palette": palette, 
        "class_to_color": class_to_color,
        "label_to_class": label_to_class,
        "label_to_color": label_to_color,
    }
