# Style Guide for Project Beginn

This document provides coding conventions for the Python code in the Project Begin repository. We've adopted these guidelines to maintain the project's codebase in a readable, consistent, and clean state. By contributing, you agree to follow this style guide in your submissions.

## Code Layout and Formatting

### Indentation

- Use 4 spaces per indentation level.

### Line Length

- Limit all lines to a maximum of 119 characters for code and 112 for comments and docstrings.

### Imports

- Imports should be on separate lines and grouped in the following order:
    1. Standard library imports.
    2. Related third-party imports.
    3. Local application/library-specific imports.

### Whitespace

- Use whitespace in expressions and statements as sparingly as possible.
- No extra spaces inside parentheses or brackets.

### Comments

- Comments should be complete sentences. If a comment is a phrase or sentence, its first word should be capitalized, unless it is an identifier that begins with a lower case letter.
- Use inline comments sparingly.

### Docstrings43

- All public modules, functions, classes, and methods should have docstrings.
- Docstrings should follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Naming Conventions

### Variables, Functions & Classes

Should follow the `CamelCase` naming convention. Folders, where used to group common pieces of code together
should be `snake_case`

### Constants

- Constants should be defined on a module level and written in all capital letters with underscores separating words.

## Programming Recommendations

### Use `is` or `is not` for Singleton Objects

- Use `is` or `is not` for comparisons with `None` and `True` / `False`

### Accessing Dictionary Elements

- Use `dict.get(key)` or `key in dict` to check for existence of a key in a dictionary.

### Comprehensions

- Use list, dict, and set comprehensions to make your code more concise and readable when appropriate.

### Data models

- All data passed in though API routes, as responses to API routes and internally for data manipulation
*MUST* have a corresponding Pydantic model class. Any data transferred internall or received (e.g. from an external resource) 
*MUST* be validated against this model using the standard Pydantic `model_validate` function which will return
a valid model instance (if your data is valid) or throw an exception. If your class makes use of optionals, 
auto-instantiated field values or overrides the modelConfig, you must include a test for your class.

## Testing

- Write tests for new code.
- Use the `pytest` framework.
- Follow the test naming conventions and structure tests logically.
- There is no need to test code in 3rd party libraries, just how your code interacts with it.

## Documentation

- Update the README.md with any changes in the API or required steps for setting up and running your code.
- Comment your code where necessary to explain complex or non-obvious parts.

This guide is not exhaustive but should serve as a basis for writing clean, readable, and consistent Python code. We encourage contributors to read the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) for a more in-depth explanation of the Python style guidelines.
