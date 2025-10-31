# Murder Model

## Henry Morgan

# Objective

- To identify a user's chances of getting murdered in a certain area based on previous homicide data.
- Target Audience: City governments for development.
- Classification problem:
  - Takes a variety of data
  - Categorizes into their likelihood of survival in each area.

# Similar / Past Works

- **Crime predictions**: An ML using regression to predict crime based on crime datasets, consisting a variety of text, images, and videos to address gaps in early detection systems.
- **Murder Accountability Project**: An ongoing project run by a non-profit organization to account for unaccounted homicides in the United States.

# Definitions

- Data Object: the user
- Some Attributes:
  - Demographics of both the user and previous victims.
  - Previous Victim Demographics
  - Locations of past homicides
  - Relationships between repeated offenders and victims.
- Target Variable: Whether the user is a victim or not.

# Limitations

- Data Acquisition (current data set dates from 1980-2014).
- Ethical considerations on other attributes to use:
  - Demographics of violators (Model needs to be inclusive).
  - May require more external data sources, but it may not imply causation from correlation.
    - Crime rates per city
    - Population statistics
