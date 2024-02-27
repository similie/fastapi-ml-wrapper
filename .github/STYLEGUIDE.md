# Style Guide for Project Beginn

This document provides coding conventions for the Python code in the Project Begin repository. We've adopted these guidelines to maintain the project's codebase in a readable, consistent, and clean state. By contributing, you agree to follow this style guide in your submissions.

Please do:
1. Write clean code
2. Write tests
3. Use type annotations
4. Prefer one code file per concept
5. Cut down on docstrings by making use of sensible function and parameter names


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
- No extra spaces inside parentheses or brackets, except between key:value pairs, e.g. ```myJson = {"field1": "value1", "field2": "value2"}```

### Comments

- Comments should be complete sentences. If a comment is a phrase or sentence, its first word should be capitalized, unless it is an identifier that begins with a lower case letter.
- Use inline comments sparingly.
- Complex use-cases should include example(s) in markdown format

### Docstrings

- All public modules, functions, classes, and methods should have docstrings.
- Docstrings should follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- All private/internal methods (classes, functions) should have docstrings unless their names (and parameter names) make their use self-evident. E.g. 
```
def numad(x,y):
    '''
    Adds x to y and returns the result
    '''
    [code body]

def addNumbers(x: int, y: int) -> int:
    [code body]
```

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
__MUST__ have a corresponding Pydantic model class. Any data transferred internally or received (e.g. from an external resource) 
__MUST__ be validated against this model. Either by instantiating a new class from input parameters or using the standard Pydantic `model_validate(jsonValue)` function [which will return
a valid model instance - if your data is valid - or throw an exception]. If your class makes use of optionals, 
auto-instantiated field values or overrides the modelConfig, please include a test for your class and any code that relies on it.

When using these data structures in your code as function parameters, ensure that they are correctly typed.

## Testing

- Write tests for new code.
- Use the `pytest` framework. 
- Follow the test naming conventions and structure tests logically.
- There is no need to test code in 3rd party libraries, just how your code interacts with it.

Note. Async methods can be tested with `asyncio`. Import `pytest` at the head of your test file and add the following decorator to your async method:
```
@pytest.mark.asyncio
async def test_my_function():
    result = await myLongRunningFunction()
    assert result is not None
```

## Documentation

- Update the README.md with any changes in the API or required steps for setting up and running your code.
- Comment your code where necessary to explain complex or non-obvious parts.

This guide is not exhaustive but should serve as a basis for writing __clean__, readable, and consistent Python code. We encourage contributors to read the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) for a more in-depth explanation of the Python style guidelines.
