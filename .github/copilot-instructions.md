# GitHub Copilot Instructions

This document provides instructions and guidelines for GitHub Copilot to help generate better code suggestions and maintain consistency across the project that also compliance with codebase context specifications (CCS) v1.1.0.

## Project Overview

This project is aimed to create a machine learning model for a classification problem of predicting murder, given the user's location and their demographics.

## Code Style Guidelines

### General Guidelines

- Use clear and descriptive variable names
- Follow consistent indentation (preferably spaces)
- Keep functions focused and single-purpose
- Add meaningful comments for complex logic
- Use appropriate error handling
- Follow DRY (Don't Repeat Yourself) principles

### Naming Conventions

- Use camelCase for variables and functions
- Use PascalCase for class names
- Use UPPER_SNAKE_CASE for constants
- Use descriptive, meaningful names

### Documentation

- Add JSDoc/Documentation comments for functions
- Include parameter descriptions
- Document return values
- Explain complex algorithms
- Add usage examples where appropriate

### Error Handling

- Use try-catch blocks for error-prone operations
- Provide meaningful error messages
- Log errors appropriately
- Handle edge cases

### Testing

- Write unit tests for new functions
- Include edge case testing
- Maintain test coverage
- Follow test naming conventions

## Project-Specific Guidelines

[Add any project-specific guidelines, patterns, or requirements here]

## Dependencies

List of main project dependencies and their usage context:

- [Dependency 1]: [Purpose/Usage]
- [Dependency 2]: [Purpose/Usage]

## File Structure

Describe the expected file structure and organization:

```
src/
├── components/
├── utils/
├── services/
└── tests/
```

## Common Patterns

Document any common patterns or approaches that should be followed:

- Pattern 1: [Description]
- Pattern 2: [Description]

## Performance Considerations

- Optimize resource usage
- Follow performance best practices
- Consider scalability

## Accessibility

- Follow WCAG guidelines
- Ensure keyboard navigation
- Provide appropriate ARIA labels

## Sustainable Code Practices

- Write maintainable code
- Consider future extensibility
- Document technical debt

## Version Control

- Write clear commit messages
- Follow branching strategy
- Keep commits focused and atomic

---

Note: Keep this file updated as the project evolves and new patterns or requirements emerge.
