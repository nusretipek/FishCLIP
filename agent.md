# Agent Guidelines

This file defines the general conventions, code style, and development practices to follow for all AI-assisted and manual code contributions in this project.

---

## 🧩 General Philosophy

- Write **clean**, **readable**, and **maintainable** code.
- Prefer **explicit** code over clever one-liners.
- Keep functions **focused and small** — ideally no more than 30–40 lines.
- Favor **composition over inheritance**.

---

## 🧱 Naming Conventions

- **camelCase** → for variables, functions, and parameters.  
  _Examples:_ `userProfile`, `getUserData()`
  
- **PascalCase** → for classes.  
  _Examples:_ `UserManager`, `AuthController`
  
- **UPPER_SNAKE_CASE** → for constants.  
  _Examples:_ `MAX_RETRIES`, `DEFAULT_TIMEOUT`
  
- **kebab-case** → for filenames.  
  _Examples:_ `user-service.js`, `data-fetcher.py`

---

## 🎨 Code Style

- Use **4-space indentation** (or your editor’s default if consistent).
- Always include **trailing commas** in multiline lists, objects, or arrays.
- Limit all lines to **100 characters maximum**.
- Always use **strict equality** (`===` / `!==`).
- Prefer **async/await** over callbacks or direct promise chaining.
- In Python functions, use typing to note the possible type of the arguments.
- In functions, assert the input, if needed.
- Except from the function docstring, do not use any comments in the code.
- If there are comments in the code, remove them (except function descriptions).
 
---

## 📘 Documentation

Every **public function**, **class**, or **module** must include a **docstring** or **JSDoc-style comment** describing:

- **Purpose**
- **Parameters**
- **Return type**
- **Example usage** (optional)

### 🐍 Example (Python)
```python
def getUserProfile(user_id: str) -> dict:
    """
    Fetch user profile data by ID.

    Args:
        user_id (str): The unique identifier of the user.

    Returns:
        dict: A dictionary containing user profile details.

    Example:
        profile = getUserProfile("abc123")
    """
